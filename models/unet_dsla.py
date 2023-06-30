import numpy as np
import torch
import torch.nn as nn

from models.unet import UnetDecoder, UnetEncoder


def get_dsla_output_layers(output_tensor, batch=True):
    '''Returns a list of correctly sliced DSLA tensors.
    Args:
        output_tensor: DSLA model output tensor (batch_n, 11, dim, dim)
        batch: Retains the batch dimension if 'True'
    Returns:
        list[0]: SLA (1 layer)
        list[1]: DA_mean (3 layers)
        list[2]: DA_var (3 layers)
        list[3]: DA_w (3 layers)
        list[4]: entry_pnt (1 layer)
        list[5]: exit_pnt (1 layer)
    '''
    if batch:
        outputs_sla = output_tensor[:, 0:1]
        outputs_dir_mean = output_tensor[:, 1:4]
        outputs_dir_var = output_tensor[:, 4:7]
        outputs_dir_weight = output_tensor[:, 7:10]
    else:
        outputs_sla = output_tensor[0:1]
        outputs_dir_mean = output_tensor[1:4]
        outputs_dir_var = output_tensor[4:7]
        outputs_dir_weight = output_tensor[7:10]

    return (outputs_sla, outputs_dir_mean, outputs_dir_var, outputs_dir_weight)


class UnetDSLA(nn.Module):

    def __init__(self,
                 enc_str,
                 sla_dec_str,
                 da_dec_str,
                 input_ch=2,
                 out_feat_ch=512,
                 num_angs=32):
        super(UnetDSLA, self).__init__()

        self.unet_encoder = UnetEncoder(enc_str, input_ch)

        bottleneck_ch = int(enc_str.split(',')[-1].split('x')[-1])
        self.unet_decoder_sla = UnetDecoder(enc_str, sla_dec_str,
                                            bottleneck_ch, out_feat_ch)
        self.unet_decoder_da = UnetDecoder(enc_str, da_dec_str, bottleneck_ch,
                                           out_feat_ch)

        # Output head 1 : Soft lane affordance
        self.sla_head = []
        self.sla_head.append(nn.Conv2d(out_feat_ch, 1, 1, stride=1, padding=0))
        self.sla_head.append(nn.Sigmoid())
        self.sla_head = nn.Sequential(*self.sla_head)

        # Output head 2 : Directional affordance
        self.da_head = []
        self.da_head.append(
            nn.Conv2d(out_feat_ch, num_angs, 1, stride=1, padding=0))
        self.da_head.append(nn.Softmax(dim=1))
        self.da_head = nn.Sequential(*self.da_head)

    def forward(self, x):

        x_bottleneck, enc_outs = self.unet_encoder(x)
        h_sla = self.unet_decoder_sla(x_bottleneck, enc_outs)
        h_da = self.unet_decoder_da(x_bottleneck, enc_outs)

        out_sla = self.sla_head(h_sla)
        out_da = self.da_head(h_da)

        out = torch.cat((out_sla, out_da), dim=1)

        return out


if __name__ == '__main__':

    enc_str = '2x32,2x32,2x64,2x64,2x128,2x128,2x256,2x256'
    dec_str = '1x128,1x128,1x64,1x64,1x32,1x32,1x16,1x16'
    input_ch = 5
    out_feat_ch = 32
    num_angs = 32
    sla_head_layers = 3
    da_head_layers = 3

    model = UnetDSLA(enc_str, dec_str, input_ch, out_feat_ch, num_angs,
                     sla_head_layers, da_head_layers)

    input_size = 256
    x = torch.rand((32, input_ch, input_size, input_size))
    # (B, C, H, W)
    print('x:', x.shape)

    y = model(x)
    # (B, C, H, W)
    print('y:', y.shape)
