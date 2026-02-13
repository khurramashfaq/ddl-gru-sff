#Traditional Recurrent Depth Network

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.update import BasicMultiUpdateBlock
from model.extractor import MultiBasicEncoder
from model.dilated_laplacian import compute_dilated_laplacian
from model.utils import upsample8, sampling_focusdim_fixlen, upsample_depth


autocast = torch.cuda.amp.autocast

class TradRDNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])




    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def forward(self, focal_stack, guidance=None, iters=32, test_mode=False):

        B,S,C,H,W = focal_stack.shape

        if guidance is not None:
            context_image = guidance

        else:
            mean_image = focal_stack.mean(dim=1)
            context_image = mean_image.contiguous() #(B,3,H,W)


        fv1,fv2,f3,fv4 = compute_dilated_laplacian(focal_stack)


        with autocast(enabled=self.args.mixed_precision):
            cnet_list = self.cnet(context_image, num_layers=3)


            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                        zip(inp_list, self.context_zqr_convs)]


        depth_initial = torch.zeros(B, 1, H // (2 ** self.args.n_downsample), W // (2 ** self.args.n_downsample), device=focal_stack.device)

        depth_predictions = []

        for itr in range(iters):
            depth_initial = depth_initial.detach()

            merged_fv = sampling_focusdim_fixlen(fv1, fv2, f3, fv4, factor=2 ** self.args.n_downsample)  # merging focus volumes

            with autocast(enabled=True):
                net_list, up_mask, delta_depth = self.update_block(net_list, inp_list, merged_fv, depth_initial, iter32=self.args.n_gru_layers == 3, iter16=self.args.n_gru_layers >= 2)

            depth_initial = depth_initial + delta_depth


            if test_mode and itr < iters - 1:
                continue

            if up_mask is None:
                depth_up = upsample8(depth_initial)
            else:
                depth_up = upsample_depth(depth_initial, up_mask, self.args.n_downsample)

            depth_predictions.append(depth_up)

        if test_mode:
            return depth_up

        return depth_predictions





