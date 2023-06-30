import glob
import gzip
import os
import pickle

import cv2
import numpy as np
import pytorch_lightning as pl
import scipy.special
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class PreprocBEVDataset():
    '''
    Intensity: Value interval (0,1)
    '''

    def __init__(
        self,
        abs_root_path,
        do_rotation=False,
        do_aug=False,
        get_gt_labels=False,
    ):
        self.abs_root_path = abs_root_path
        self.sample_paths = glob.glob(
            os.path.join(self.abs_root_path, '*', '*.pkl.gz'))

        self.sample_paths = [
            os.path.relpath(path, self.abs_root_path)
            for path in self.sample_paths
        ]
        self.sample_paths.sort()

        self.do_rotation = do_rotation
        self.do_aug = do_aug
        self.get_gt_labels = get_gt_labels

        self.transf_rgb = torch.nn.Sequential(
            transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5),
            transforms.GaussianBlur(3, sigma=(0.001, 2.0)),
        )

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):

        sample_path = self.sample_paths[idx]
        sample_path = os.path.join(self.abs_root_path, sample_path)
        input, label = self.read_compressed_pickle(sample_path)

        # Add auxhilary labels
        drivable = input[0:1].clone()
        label['drivable'] = drivable

        # Add label channel dims
        # label['traj_present'] = label['traj_present'].unsqueeze(0)
        # label['traj_present'] = label['traj_present'].float()

        traj = label['traj_full'].numpy().astype(float)
        kernel = np.ones((3, 3), np.uint8)
        traj = cv2.dilate(traj, kernel)
        label['traj_full'] = torch.tensor(traj, dtype=torch.float32)
        label['traj_full'] = label['traj_full'].unsqueeze(0)

        # Transform list of angles to multimodal distribution tensor
        # NOTE: Unobserved elements have uniform distribution
        num_discr = 36
        m_max = 88
        mm_ang_full_tensor = self.gen_multimodal_vonmises_distrs(
            label['angs_full'], num_discr, m_max)
        mm_ang_full_tensor = np.transpose(mm_ang_full_tensor, (2, 0, 1))
        label['mm_ang_full_tensor'] = torch.tensor(mm_ang_full_tensor)

        if self.get_gt_labels:

            gt_lanes = label['gt_lanes'].numpy().astype(float)
            gt_lanes = cv2.dilate(gt_lanes, kernel)
            label['gt_lanes'] = torch.tensor(gt_lanes, dtype=torch.float32)
            label['gt_lanes'] = label['gt_lanes'].unsqueeze(0)

            mm_gt_angs_tensor = self.gen_multimodal_vonmises_distrs(
                label['gt_angs'], num_discr, m_max)
            mm_gt_angs_tensor = np.transpose(mm_gt_angs_tensor, (2, 0, 1))
            label['mm_gt_angs_tensor'] = torch.tensor(mm_gt_angs_tensor)

        # Random rotation
        # # TODO Need fix for new multimodal angle repr.
        # if self.do_rotation:
        #     k = random.randrange(0, 4)
        #     tensor_rot = torch.rot90(tensor, k, (-2, -1))
        #     tensor_rot_ = tensor_rot.clone()
        #     if k == 1:
        #         tensor_rot[-2] = tensor_rot_[-1] * (-1)
        #         tensor_rot[-1] = tensor_rot_[-2]
        #     elif k == 2:
        #         tensor_rot[-2] = tensor_rot_[-2] * (-1)
        #         tensor_rot[-1] = tensor_rot_[-1] * (-1)
        #     elif k == 3:
        #         tensor_rot[-2] = tensor_rot_[-1]
        #         tensor_rot[-1] = tensor_rot_[-2] * (-1)
        #     tensor = tensor_rot

        # Augmentation for intensity and RGB map (to limit overfitting)
        if self.do_aug:
            # Intensity
            # Randomly samples a set of augmentations
            input_int = input[1].clone().numpy()
            input_int = self.rand_aug_int(input_int)
            input[1] = torch.tensor(input_int)
            # RGB
            input_rgb = (255 * input[2:5]).type(torch.uint8)
            input_rgb = self.transf_rgb(input_rgb)
            input[2:5] = input_rgb.float() / 255

        # Transform input value range (0, 1) --> (-1, 1)
        input = (2 * input) - 1.

        # Remove unrelated entries
        rm_keys = ['map', 'scene_idx', 'ego_global_x', 'ego_global_y']
        for rm_key in rm_keys:
            if rm_key in label.keys():
                del label[rm_key]

        # Ensure that all tensors are of the same type
        for key in label.keys():
            label[key] = label[key].type(torch.float)

        return input, label

    def rand_aug_int(self,
                     x,
                     num_augs_min=1,
                     num_augs_max=4,
                     p_cat_distr=[0.3, 0.15, 0.15, 0.4]):
        num_augs = np.random.randint(num_augs_min, num_augs_max)
        augs = np.random.choice(np.arange(4), size=num_augs, p=p_cat_distr)
        for aug_idx in augs:
            if aug_idx == 0:
                x = self.sharpen(x)
            elif aug_idx == 1:
                x = self.gaussian_blur(x)
                x = self.sharpen(x)
            elif aug_idx == 2:
                x = self.box_blur(x)
                x = self.sharpen(x)
            elif aug_idx == 3:
                x = self.scale(x)
            else:
                raise Exception('Undefined augmentation')
        x = self.normalize(x)

        return x

    def gen_multimodal_vonmises_distrs(self,
                                       angs,
                                       num_discr,
                                       vonmises_m,
                                       height=256,
                                       width=256):
        '''
        Args:
            angs: (N,3)
            num_discr: Number of elements discretizing (0, 2*pi)
            vonmises_m: Von Mises distribution concentration parameter.

        Returns:
            Tensor with multimodal von Mises distributions for labeled elements
            w. dim(num_discr, H, W)
        '''
        ang_range = np.linspace(0, 2 * np.pi, num_discr)
        vonmises_b = scipy.special.i0(vonmises_m)

        # Add angles into element-wise lists
        ang_dict = {}
        for idx in range(angs.shape[0]):
            i, j, ang = angs[idx]
            i = int(i.item())
            j = int(j.item())
            ang = ang.item()

            # Negative entries [-1, -1, -1] means end of list
            if i < 0:
                break

            # Initialize empty array for first encountered element
            if (i, j) not in ang_dict.keys():
                ang_dict[(i, j)] = []

            # Add angle to multimodal distribution for element
            ang_dict[(i, j)].append(ang)

        # Initialize uniform distribution tensor
        distr_tensor = np.ones((height, width, num_discr)) / num_discr

        # Create multimodal von Mises distribution for elements
        for elem in ang_dict.keys():
            i, j = elem

            num_angs = len(ang_dict[(i, j)])

            mm_distr = np.zeros_like(ang_range)

            for mode_idx in range(num_angs):

                mode_ang = ang_dict[(i, j)][mode_idx]

                distr = np.exp(vonmises_m * np.cos(ang_range - mode_ang))
                distr /= (2.0 * np.pi * vonmises_b)

                # Preserve significance of each mode independent of frequency
                mm_distr = np.maximum(distr, mm_distr)

            # Normalize distribution
            mm_distr /= self.integrate_distribution(mm_distr, ang_range)

            distr_tensor[i, j] = mm_distr

        return distr_tensor

    @staticmethod
    def integrate_distribution(dist, dist_range):
        '''Integrate a distribution using the trapezoidal approximation rule.

            Args:
                dist: Distribution values in 1D array.
                dist_range: Distrbution range in 1D array.

            Returns:
                Integration sum as float.
            '''
        N = dist.shape[0]
        integ_sum = 0.0
        for i in range(N - 1):
            partion_range = dist_range[i + 1] - dist_range[i]
            dist_val = dist[i] + dist[i + 1]
            integ_sum += partion_range * dist_val / 2.0

        return integ_sum

    @staticmethod
    def read_compressed_pickle(path):
        try:
            with gzip.open(path, "rb") as f:
                pkl_obj = f.read()
                obj = pickle.loads(pkl_obj)
                return obj
        except IOError as error:
            print(error)

    @staticmethod
    def sharpen(array):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(array, -1, kernel)

    @staticmethod
    def gaussian_blur(array):
        kernel = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        return cv2.filter2D(array, -1, kernel)

    @staticmethod
    def box_blur(array):
        kernel = (1 / 9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        return cv2.filter2D(array, -1, kernel)

    @staticmethod
    def scale(array, thresh_min=0.25, thresh_max=0.75):
        scale = np.random.normal(loc=1., scale=0.2)
        scale = max(scale, thresh_min)
        scale = min(scale, thresh_max)
        return scale * array

    @staticmethod
    def normalize(array):
        mask = array > 1.
        array[mask] = 1.
        mask = array < 0.
        array[mask] = 0.
        return array


class BEVDataPreprocModule(pl.LightningDataModule):

    def __init__(
        self,
        train_data_dir: str = "./",
        val_data_dir: str = "./",
        test_data_dir: str = "./",
        batch_size: int = 128,
        num_workers: int = 0,
        persistent_workers=True,
        do_rotation: bool = False,
        do_aug: bool = False,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.bev_dataset_train = PreprocBEVDataset(
            self.train_data_dir,
            do_rotation=do_rotation,
            do_aug=do_aug,
        )
        # NOTE Loads GT lane map for evaluation
        self.bev_dataset_val = PreprocBEVDataset(self.val_data_dir,
                                                 get_gt_labels=True)
        self.bev_dataset_test = PreprocBEVDataset(self.test_data_dir,
                                                  get_gt_labels=True)

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.bev_dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.bev_dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            self.bev_dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )


if __name__ == '__main__':
    '''
    For visualizing dataset tensors.
    '''

    from viz.viz_dataset import viz_dataset_sample

    batch_size = 1

    ###############################
    #  Load preprocessed dataset
    ###############################

    bev = BEVDataPreprocModule('bev_nuscenes_256px_v01_job01_rl_preproc',
                               'bev_nuscenes_256px_v01_job01_rl_preproc',
                               'bev_nuscenes_256px_v01_job01_rl_preproc',
                               batch_size,
                               do_rotation=False,
                               do_aug=False)

    dataloader = bev.train_dataloader(shuffle=False)

    for idx, batch in enumerate(dataloader):

        inputs, labels = batch

        # Transform input value range (-1, 1) --> (0, 1)
        inputs = 0.5 * (inputs + 1)

        # Remove batch index in each tensor
        inputs = inputs[0]
        for key in labels.keys():
            labels[key] = labels[key][0]

        viz_dataset_sample(inputs, labels)
