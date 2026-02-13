import torch
import torch.nn.functional as F



def compute_dilated_laplacian(focal_stack):
    B, S, C, H, W = focal_stack.shape

    fs_flat = focal_stack.view(B * S, C, H, W)

    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=focal_stack.device).view(1, C, 1, 1)
    fs_gray = (fs_flat * weights).sum(dim=1, keepdim=True)  # (B*S, 1, H, W)


    k1_3x3 = torch.tensor([[0., 0., 0.],
                           [1., -2., 1.],
                           [0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # x-axis

    k2_3x3 = torch.tensor([[0., 1., 0.],
                           [0., -2., 0.],
                           [0., 1., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # y-axis

    k3_3x3 = torch.tensor([[1., 0., 0.],
                           [0., -2., 0.],
                           [0., 0., 1.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # diagonal 1

    k4_3x3 = torch.tensor([[0., 0., 1.],
                           [0., -2., 0.],
                           [1., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # diagonal 2

    # 5x5 kernels with dilation=2
    k1_5x5 = torch.tensor([[0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [1., 0., -2., 0., 1.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # x-axis

    k2_5x5 = torch.tensor([[0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., -2., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # y-axis

    k3_5x5 = torch.tensor([[1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., -2., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 1.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # diagonal 1

    k4_5x5 = torch.tensor([[0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., -2., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [1., 0., 0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # diagonal 2

    # 7x7 kernels with dilation=3
    k1_7x7 = torch.tensor([[0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [1., 0., 0., -2., 0., 0., 1.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # x-axis

    k2_7x7 = torch.tensor([[0., 0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., -2., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(0)  # y-axis

    k3_7x7 = torch.tensor([[1., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., -2., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 1.]], device=focal_stack.device).unsqueeze(0).unsqueeze(
        0)  # diagonal 1

    k4_7x7 = torch.tensor([[0., 0., 0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., -2., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [1., 0., 0., 0., 0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(
        0)  # diagonal 2

    # 9x9 kernels with dilation=4
    k1_9x9 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [1., 0., 0., 0., -2., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(
        0)  # x-axis

    k2_9x9 = torch.tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., -2., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 1., 0., 0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(
        0)  # y-axis

    k3_9x9 = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., -2., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 1.]], device=focal_stack.device).unsqueeze(0).unsqueeze(
        0)  # diagonal 1

    k4_9x9 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., -2., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [1., 0., 0., 0., 0., 0., 0., 0., 0.]], device=focal_stack.device).unsqueeze(0).unsqueeze(
        0)  # diagonal 2


    fv_3x3 = torch.abs(F.conv2d(fs_gray, k1_3x3, padding=1)) + \
             torch.abs(F.conv2d(fs_gray, k2_3x3, padding=1)) + \
             torch.abs(F.conv2d(fs_gray, k3_3x3, padding=1)) + \
             torch.abs(F.conv2d(fs_gray, k4_3x3, padding=1))

    fv_5x5 = torch.abs(F.conv2d(fs_gray, k1_5x5, padding=2)) + \
             torch.abs(F.conv2d(fs_gray, k2_5x5, padding=2)) + \
             torch.abs(F.conv2d(fs_gray, k3_5x5, padding=2)) + \
             torch.abs(F.conv2d(fs_gray, k4_5x5, padding=2))

    fv_7x7 = torch.abs(F.conv2d(fs_gray, k1_7x7, padding=3)) + \
             torch.abs(F.conv2d(fs_gray, k2_7x7, padding=3)) + \
             torch.abs(F.conv2d(fs_gray, k3_7x7, padding=3)) + \
             torch.abs(F.conv2d(fs_gray, k4_7x7, padding=3))

    fv_9x9 = torch.abs(F.conv2d(fs_gray, k1_9x9, padding=4)) + \
             torch.abs(F.conv2d(fs_gray, k2_9x9, padding=4)) + \
             torch.abs(F.conv2d(fs_gray, k3_9x9, padding=4)) + \
             torch.abs(F.conv2d(fs_gray, k4_9x9, padding=4))


    fv_3x3 = fv_3x3.view(B, S, 1, H, W)
    fv_5x5 = fv_5x5.view(B, S, 1, H, W)
    fv_7x7 = fv_7x7.view(B, S, 1, H, W)
    fv_9x9 = fv_9x9.view(B, S, 1, H, W)

    return fv_3x3, fv_5x5, fv_7x7, fv_9x9


