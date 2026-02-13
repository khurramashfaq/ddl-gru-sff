import os
import torch
import logging
import matplotlib.pyplot as plt
from datareader.FoD import Focus_on_Defocus
from torch.utils.data import DataLoader
from utils.metric_utils import compute_sample_metrics, save_metrics_to_csv
from model.TradRDNet import TradRDNet
from config import get_config


def save_test_depths(model, test_loader, cfg):
    model.eval()
    model.to(cfg.device)
    output_dir = f'test_results/{cfg.experiment_name}'
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []

    with torch.no_grad():
        for i, (focal_stack, depth_gt) in enumerate(test_loader):
            focal_stack, depth_gt = focal_stack.to(cfg.device), depth_gt.to(cfg.device)
            depth_map = model(focal_stack, guidance=None, iters=cfg.update_iters, test_mode=True)

            metrics = compute_sample_metrics(depth_map, depth_gt)
            all_metrics.extend(metrics)

            depth_map_np = depth_map.squeeze().cpu().numpy()
            plt.imsave(f'{output_dir}/depth_computed_{i}.png', depth_map_np, cmap='viridis')

        avg_metrics_str=save_metrics_to_csv(all_metrics, output_dir, cfg.experiment_name)
        logging.info(f"\nAverage Test Metrics:\n{avg_metrics_str}")


if __name__ == '__main__':
    config_args = get_config('FoD')
    config_args.experiment_name = 'testckpt_FoD'
    test_loader = DataLoader(Focus_on_Defocus(config_args.data_path, 'test'), batch_size=1, shuffle=False)

    model = TradRDNet(config_args)
    pth_path = r".\checkpoints\ckpt_FoD.pth"
    model.load_state_dict(torch.load(pth_path, map_location=config_args.device))
    model.to(config_args.device)

    save_test_depths(model, test_loader, config_args)

    print(f"Finished. Saved outputs in test_results/{config_args.experiment_name}")