import torch
import torch.nn as nn
import torch.nn.functional as F




def sampling_focusdim_fixlen(fv1, fv2, fv3, fv4, desired_length=15, factor=8):
    def resample(fv):
        B, S, C, H, W = fv.shape
        idx = torch.linspace(0, S - 1, desired_length).round().long()
        return torch.stack([fv[b, idx] for b in range(B)], dim=0)
    r1, r2, r3, r4 = (resample(x) for x in (fv1, fv2, fv3, fv4))  # each (B,15,C,H,W)
    stacked = torch.stack([r1, r2, r3, r4], dim=2)               # (B,15,4,C,H,W)
    restacked = stacked.reshape(stacked.size(0), -1, stacked.size(3), stacked.size(4), stacked.size(5)) ## B,4S,1,H/8,W/8
    downsampled = F.avg_pool3d(restacked, kernel_size=(1, factor, factor), stride=(1, factor, factor))  # B,4S,1,H/8,W/8

    return downsampled.squeeze(2)




def upsample8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)



def upsample_depth(depth, mask, n_downsample):
    """ Upsample depth using convex combination """

    N, D, H, W = depth.shape  # [1, 1, 32, 32]
    factor = 2 ** n_downsample # factor = 8 if ndownsample=3
    mask = mask.view(N, 1, 9, factor, factor, H, W) # [1, 1, 9, 8, 8, 32, 32
    mask = torch.softmax(mask, dim=2)
    up_flow = F.unfold(factor * depth, [3, 3], padding=1) # [1, 9, 32*32]
    up_flow = up_flow.view(N, D, 9, 1, 1, H, W)  # [1, 1, 9, 1, 1, 32, 32]
    up_flow = torch.sum(mask * up_flow, dim=2)  # [1, 1, 8, 8, 32, 32]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, D, factor * H, factor * W)


