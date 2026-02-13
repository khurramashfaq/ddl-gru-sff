import torch


def compute_sequential_depth_loss(depth_predictions, depth_gt, alpha=0.9):

    total_loss = 0.0
    N = len(depth_predictions)

    for i, depth_pred in enumerate(depth_predictions):

        eps = 1e-8
        depth_pred = torch.nan_to_num(depth_pred, nan=eps)
        depth_gt = torch.nan_to_num(depth_gt, nan=eps)

        weight = alpha ** (N - i - 1)
        total_loss += weight * torch.nn.functional.mse_loss(depth_pred, depth_gt)


    return total_loss