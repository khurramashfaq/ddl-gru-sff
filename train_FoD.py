import os
import torch
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from datareader.FoD import Focus_on_Defocus
from torch.utils.data import DataLoader
from loss.sequence_loss import compute_sequential_depth_loss
from utils.metric_utils import compute_sample_metrics, save_metrics_to_csv
from torch.cuda.amp import GradScaler
from model.TradRDNet import TradRDNet
from config import get_config

def train_and_test_model(cfg):

    experiment_dir = f'test_results/{cfg.experiment_name}'
    os.makedirs(experiment_dir, exist_ok=True)

    loss_log = os.path.join(experiment_dir, f"Loss_{cfg.experiment_name}.csv")
    with open(loss_log, 'w') as f:
        f.write("Epoch,MeanLoss\n")

    log_file_path = os.path.join(experiment_dir, f"Log_{cfg.experiment_name}.log")
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler()
        ])

    train_loader = DataLoader(Focus_on_Defocus(cfg.data_path, 'train'), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(Focus_on_Defocus(cfg.data_path, 'test'), batch_size=1, shuffle=False)

    model = TradRDNet(cfg)
    print(f'Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if cfg.restore_ckpt:
        checkpoint = torch.load(cfg.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model = model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    best_loss = float('inf')
    scaler = GradScaler(enabled=cfg.mixed_precision)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0
        pbar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}/{cfg.epochs}', total=len(train_loader))
        for batch_idx, (focal_stack, depth_gt) in pbar:
            optimizer.zero_grad()

            focal_stack, depth_gt = focal_stack.to(cfg.device), depth_gt.to(cfg.device)

            depth_preds = model(focal_stack, guidance=None, iters=cfg.update_iters, test_mode=False)

            loss = compute_sequential_depth_loss(depth_preds, depth_gt)

            running_loss += loss.item()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'Meanloss': f'{running_loss / (batch_idx + 1):.4f}'})

        m_loss = running_loss / len(train_loader)
        with open(loss_log, 'a') as f:
            f.write(f"{epoch + 1},{m_loss:.6f}\n")
        grad_norms = [param.grad.norm(2).item() for param in model.parameters() if param.grad is not None]

        logging.info("Epoch %d:, LR : %.5f, MaxGradNorm: %.5f, MeanLoss: %.5f", epoch+1, optimizer.param_groups[0]['lr'], max(grad_norms), m_loss)

        scheduler.step()

        if epoch % (5) == 0:
            logging.info(f"Saving depth maps for epoch {epoch}")
            save_test_depths(model, test_loader, epoch, cfg)
            output_dir = f'test_results/{cfg.experiment_name}/Ep_{epoch}'
            last_epoch_model_path = f'{output_dir}/{cfg.experiment_name}_Ep_{epoch}.pth'
            torch.save(model.state_dict(), last_epoch_model_path)

        if m_loss < best_loss:
            best_loss = m_loss
            best_model_path = f'{experiment_dir}/{cfg.experiment_name}_best_model.pth'
            torch.save(model.state_dict(), best_model_path)


def save_test_depths(model, test_loader, epoch, cfg):
    model.eval()
    model.to(cfg.device)
    all_metrics = []
    output_dir = f'test_results/{cfg.experiment_name}/Ep_{epoch}'
    os.makedirs(output_dir, exist_ok=True)

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
    config_args.experiment_name = input("Enter name for this experiment: ")
    train_and_test_model(config_args)
