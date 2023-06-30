import os

import cv2
import matplotlib as mpl

mpl.use('agg')  # Must be before pyplot import to avoid memory leak
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from datamodule_preproc import PreprocBEVDataset
from eval.eval_dir_accuracy import eval_dir_acc
from eval.eval_f1_score import eval_f1_score
from eval.eval_iou import eval_iou
from graph_inference.graph_func import comp_entry_exits
from graph_inference.max_likelihood_graph import find_max_likelihood_graph
from losses.da_model_free_kl_div import loss_da_kl_div
from losses.da_nll import eval_da_nll
from losses.sla_balanced_ce import loss_sla_balanced_ce
from losses.sla_nll import eval_sla_nll
from models.unet_dsla import UnetDSLA
from viz.viz_dense import visualize_dense_softmax, visualize_dir_label


class DSLAModel(pl.LightningModule):
    '''
    '''

    def __init__(
        self,
        sla_alpha,
        lr,
        weight_decay,
        base_channels,
        enc_str,
        sla_dec_str,
        da_dec_str,
        dropout_prob,
        print_interval,
        checkpoint_interval,
        batch_size,
        input_ch,
        out_feat_ch,
        num_angs,
        sla_head_layers,
        da_head_layers,
        num_workers,
        train_data_dir,
        val_data_dir,
        test_data_dir,
        viz_size_per_fig,
        viz_dir,
        optimizer,
        test_batch_idxs,
        test_start_batch_idx,
        viz_size,
        output_test_dir,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.sla_alpha = sla_alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.base_channels = base_channels
        self.dropout_prob = dropout_prob
        self.print_interval = print_interval
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.viz_size_per_fig = viz_size_per_fig
        self.optimizer = optimizer
        if test_batch_idxs is not None:
            self.test_batch_idxs = test_batch_idxs
        else:
            self.test_batch_idxs = []
        self.test_start_batch_idx = test_start_batch_idx
        self.viz_size = viz_size
        self.output_test_dir = output_test_dir

        # Set of choosen samples for visualization
        self.viz_dataset = PreprocBEVDataset(viz_dir, get_gt_labels=True)

        ###########
        #  Model
        ###########
        self.model = UnetDSLA(enc_str, sla_dec_str, da_dec_str, input_ch,
                              out_feat_ch, num_angs)

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        else:
            raise IOError(f'Invalid optimizer ({self.optimizer})')

        power = 0.9
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, self.trainer.max_epochs, power)
        return [optimizer], [scheduler]

    def dsla_objective(self, output, traj_label, ang_label, drivable):
        ''' Computes a multi-objective loss to traing a model

        Output tensor shape: (minibatch_idx, output layers, n, n)

        Output layers:
            --------------------------------------------------
            [0]    Soft lane affordance (1 layer)
            --------------------------------------------------
            [1]  Directional mean 1 (3 layers) # 6 layers
            [2]  Directional mean 2
            [3]  Directional mean 3
            --------------------------------------------------
            [4]    Directional var 1 (3 layers)
            [5]    Directional var 2
            [6]    Directional var 2
            --------------------------------------------------
            [7]   Directional weight 1 (3 layers)
            [8]   Directional weight 2
            [9]   Directional weight 3
            --------------------------------------------------
        '''
        output_sla = output[:, 0:1]
        output_da = output[:, 1:37]

        # Remove non-road elements
        mask = (drivable == 0)
        # Compute drivable elements for each batch
        drivable_N = torch.sum(~mask, dim=(1, 2, 3), keepdim=True)

        # Soft Lane Affordance loss [batch_n, 1, n, n]
        loss_sla, loss_l1, loss_ce = loss_sla_balanced_ce(
            output_sla, traj_label, self.sla_alpha, drivable_N)

        # Directional Affordance loss
        loss_da = loss_da_kl_div(output_da, ang_label)

        loss = 300 * loss_sla + loss_da

        return loss, loss_sla.item(), loss_da.item(), loss_l1, loss_ce

    def eval_objective(self, output, traj_label, ang_label, drivable):
        '''
        '''
        output_sla = output[:, 0:1]
        output_da = output[:, 1:37]

        mask = (drivable == 0)
        drivable_N = torch.sum(~mask, dim=(1, 2, 3))

        sla_nll = eval_sla_nll(output_sla, traj_label, drivable_N)
        da_nll = eval_da_nll(output_da, ang_label)

        return sla_nll.item(), da_nll.item()

    def forward(self, x):
        y = self.model.forward(x)
        return y

    def training_step(self, batch, batch_idx):

        input, labels = batch

        traj_label = labels['traj_full']
        mm_ang_tensor = labels['mm_ang_full_tensor']
        drivable = labels['drivable']

        output_tensor = self.forward(input)

        lst = self.dsla_objective(
            output_tensor,
            traj_label,
            mm_ang_tensor,
            drivable,
        )
        loss, loss_sla, loss_da, loss_sla_l1, loss_sla_ce = lst

        self.log_dict({
            'lr': self.optimizers().param_groups[0]["lr"],
            'train_loss': loss,
            'train_loss_sla': loss_sla,
            'train_loss_da': loss_da,
            'train_loss_sla_l1': loss_sla_l1,
            'train_loss_sla_ce': loss_sla_ce,
        })

        return loss

    def validation_step(self, batch, batch_idx):

        input, labels = batch

        traj_label = labels['gt_lanes']
        mm_ang_tensor = labels['mm_gt_angs_tensor']
        drivable = labels['drivable']

        output = self.forward(input)

        sla_nll, da_nll = self.eval_objective(
            output,
            traj_label,
            mm_ang_tensor,
            drivable,
        )

        self.log_dict({
            'val_sla_nll': sla_nll,
            'val_da_nll': da_nll,
        },
                      sync_dist=True)

    def validation_epoch_end(self, val_step_outputs):

        # Load a static set of visualization examples
        vizs = []
        num_samples = len(self.viz_dataset)
        for sample_idx in range(num_samples):
            input, label = self.viz_dataset[sample_idx]
            viz, _, _, _ = self.viz_output(input, label)
            vizs.append(viz)

        # Arrange viz side-by-side
        vizs = np.concatenate(vizs, axis=1)

        plt.figure(figsize=((num_samples * self.viz_size_per_fig,
                             self.viz_size_per_fig)))
        plt.imshow(vizs)
        plt.tight_layout()

        self.logger.experiment.add_figure('viz',
                                          plt.gcf(),
                                          global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        '''
        '''
        input, label = batch

        if input.shape[0] != 1:
            raise IOError('Test function requires batch size = 1')

        # Skip batches before starting idx
        if batch_idx < self.test_start_batch_idx:
            return

        # Skip unlisted batches if any are specified
        if len(self.test_batch_idxs) != 0:
            if batch_idx not in self.test_batch_idxs:
                return

        if not os.path.isdir(self.output_test_dir):
            os.makedirs(self.output_test_dir)

        filename = f'viz_{str(batch_idx).zfill(3)}.png'
        filepath = os.path.join(self.output_test_dir, filename)

        ################################################
        #  Negative log likelihood evaluation metrics
        ################################################
        traj_label = label['gt_lanes']
        mm_ang_tensor = label['mm_gt_angs_tensor']
        drivable = label['drivable']

        output = self.forward(input)

        sla_nll, da_nll = self.eval_objective(
            output,
            traj_label,
            mm_ang_tensor,
            drivable,
        )

        #################################################
        #  Maximum likelihood graph evaluation metrics
        #################################################
        # Remove batch indices
        input_no_b = input[0]
        label_no_b = {key: value[0] for key, value in zip(label.keys(),
                                                          label.values())}
        rgb_viz, iou, f1_score, dir_acc = self.viz_output(input_no_b,
                                                          label_no_b,
                                                          do_graph=True)
        plt.figure(figsize=((self.viz_size, self.viz_size)))
        plt.imshow(rgb_viz)
        plt.tight_layout()
        plt.savefig(filepath)

        # iou = 0.
        # f1_score = 0.
        # dir_acc = 0.

        eval_file = os.path.join(self.output_test_dir, 'eval.txt')
        if os.path.isfile(eval_file):
            mode = 'a'
        else:
            mode = 'w'
        with open(eval_file, mode) as f:
            f.write(
                f'{batch_idx},{sla_nll},{da_nll},{iou},{f1_score},{dir_acc}\n')

        return sla_nll, da_nll, iou, f1_score, dir_acc

    def test_epoch_end(self, test_step_outputs):

        sla_nlls = []
        da_nlls = []
        ious = []
        f1_scores = []
        dir_accs = []
        for out in test_step_outputs:
            sla_nll, da_nll, iou, f1_score, dir_acc = out
            sla_nlls.append(sla_nll)
            da_nlls.append(da_nll)
            ious.append(iou)
            f1_scores.append(f1_score)
            dir_accs.append(dir_acc)

        sla_nll_mean = np.mean(sla_nlls)
        da_nll_mean = np.mean(da_nlls)
        iou_mean = np.mean(ious)
        f1_scores_mean = np.mean(f1_scores)
        dir_accs_mean = np.mean(dir_accs)

        print('')
        print('\nEvaluation result')
        print(f'\tsla_nll_mean: {sla_nll_mean:.3f}')
        print(f'\tda_nll_mean: {da_nll_mean:.3f}')
        print(f'\tiou_mean: {iou_mean:.3f}')
        print(f'\tf1_scores_mean: {f1_scores_mean:.3f}')
        print(f'\tdir_accs_mean: {dir_accs_mean:.3f}')
        print('')

    def viz_output(self,
                   input: torch.tensor,
                   label: dict,
                   do_graph: bool = False,
                   use_cuda: bool = True) -> np.array:
        '''
        Args:
            input: (5, 256, 256)
            label: Dict with tensor
            do_graph: Overlay inferred graph if True

        Returns:
            RGB image (1280,1280,3)
        '''
        gt_lanes = label['gt_lanes']
        mm_gt_angs_tensor = label['mm_gt_angs_tensor']
        drivable = label['drivable']

        input = input.unsqueeze(0)
        if use_cuda:
            input = input.cuda()
        with torch.no_grad():
            output = self.forward(input)
        output = output[0].cpu().numpy()
        input = input[0].cpu().numpy()
        mm_gt_angs_tensor = mm_gt_angs_tensor.cpu()  # [0]
        drivable = drivable[0].cpu()

        output_sla = output[0:1]
        output_da = output[1:37]

        # Remove non-drivable region
        mask = (drivable == 1).numpy()
        output_sla[0][mask == 0] = 0.0

        # Dense visualization
        drivable_in = input[0]
        markings = input[1]
        context = 0.1 * drivable_in + 0.9 * markings
        context = (255 * context).astype(np.int8)
        rgb_viz = visualize_dense_softmax(context, output_sla[0], output_da,
                                          None)

        # Overlay direction label (NOT predicted direction)
        rgb_viz = visualize_dir_label(rgb_viz, mm_gt_angs_tensor)

        # For skipping RGB visualization (to save time)
        # rgb_viz = np.zeros((1280, 1280, 3), dtype=np.uint8)

        # Overlay inferred road lane network graph
        if do_graph:
            # Downscale images for search function
            ds_size = 128
            out_sla_ds = cv2.resize(output_sla[0], (ds_size, ds_size),
                                    interpolation=cv2.INTER_LINEAR)
            num_dirs = output_da.shape[0]
            out_da_ds = np.zeros((num_dirs, ds_size, ds_size))
            for dir_n in range(num_dirs):
                out_da_ds[dir_n] = cv2.resize(output_da[dir_n],
                                              (ds_size, ds_size),
                                              interpolation=cv2.INTER_NEAREST)
            # Normalize prob values
            out_da_ds /= np.sum(out_da_ds, axis=(0))
            entry_paths, connecting_pnts, exit_paths = comp_entry_exits(
                out_sla_ds, out_da_ds)

            paths = find_max_likelihood_graph(output_sla[0],
                                              output_da,
                                              entry_paths,
                                              connecting_pnts,
                                              exit_paths,
                                              num_samples=1000,
                                              num_pnts=100)

            ##########################
            #  Numerical evaluation
            ##########################
            iou = eval_iou(paths, gt_lanes[0, 0].cpu().numpy())
            f1_score = eval_f1_score(paths, gt_lanes[0, 0].cpu().numpy(),
                                     drivable[0].numpy())
            dir_acc = eval_dir_acc(output_da, mm_gt_angs_tensor.numpy())

            # Transform path coordinates to image frame
            # 10 x (256 --> 128)
            scale_factor = 10
            paths = [scale_factor * (path // 2) for path in paths]

            t = 3
            l = 0.5
            color = (0, 102, 204)

            for path in paths:
                # pnts = np.expand_dims(path, 1)  # (N, 1, 2)
                pnts = path.astype(np.int32)
                pnts = pnts.reshape((-1, 1, 2))
                rgb_viz = cv2.polylines(rgb_viz, [pnts],
                                        isClosed=False,
                                        color=color,
                                        thickness=t)
                rgb_viz = cv2.arrowedLine(rgb_viz,
                                          pnts[-2, 0],
                                          pnts[-1, 0],
                                          color=color,
                                          thickness=t,
                                          tipLength=l)
        else:
            iou = None
            f1_score = None
            dir_acc = None

        return rgb_viz, iou, f1_score, dir_acc

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DSLAModel')
        parser.add_argument('--sla_alpha', type=float, default=1.)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument(
            '--enc_str',
            type=str,
            default='2x32,2x32,2x64,2x64,2x128,2x128,2x256,2x256')
        parser.add_argument(
            '--sla_dec_str',
            type=str,
            default='2x256,2x256,2x128,2x128,2x64,2x64,2x32,2x32')
        parser.add_argument(
            '--da_dec_str',
            type=str,
            default='2x256,2x256,2x128,2x128,2x64,2x64,2x32,2x32')
        parser.add_argument('--base_channels', type=int, default=64)
        parser.add_argument('--dropout_prob', type=float, default=0)
        parser.add_argument('--print_interval', type=int, default=100)
        parser.add_argument('--checkpoint_interval', type=int, default=1000)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--input_ch', type=int, default=5)
        parser.add_argument('--out_feat_ch', type=int, default=512)
        parser.add_argument('--num_angs', type=int, default=32)
        parser.add_argument('--sla_head_layers', type=int, default=3)
        parser.add_argument('--da_head_layers', type=int, default=3)
        parser.add_argument('--viz_size_per_fig', type=int, default=12)
        parser.add_argument('--viz_dir', type=str)
        parser.add_argument('--optimizer',
                            type=str,
                            default='adam',
                            help='adam|sgd')
        parser.add_argument('--test_batch_idxs',
                            type=int,
                            nargs='+',
                            help='11 12 etc')
        parser.add_argument('--test_start_batch_idx', type=int, default=0)
        parser.add_argument('--viz_size',
                            type=int,
                            default=12,
                            help='Size of output viz image')
        parser.add_argument('--output_test_dir', type=str, default='.')
        return parent_parser

if __name__ == '__main__':
    from argparse import ArgumentParser, BooleanOptionalAction

    from datamodule_preproc import BEVDataPreprocModule

    torch.set_float32_matmul_precision('high')

    parser = ArgumentParser()
    parser.add_argument('--train_data_dir', type=str)
    parser.add_argument('--val_data_dir', type=str)
    parser.add_argument('--test_data_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--do_augmentation', action=BooleanOptionalAction)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--do_test', action=BooleanOptionalAction)
    # Add program level args
    # Add model speficic args
    parser = DSLAModel.add_model_specific_args(parser)
    # Add all the vailable trainer option to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    model = DSLAModel(**dict_args)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # To save every checkpoint
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="val_sla_nll",
        filename="checkpoint_{epoch:02d}",
    )
    # Ref: https://github.com/Lightning-AI/lightning/issues/3648
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[checkpoint_callback])

    datamodule = BEVDataPreprocModule(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        test_data_dir=args.test_data_dir,
        batch_size=args.batch_size,
        do_rotation=False,
        do_aug=args.do_augmentation,
        num_workers=args.num_workers,
    )

    if args.do_test:
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule)
