import pandas as pd
import numpy as np
import logging
from utils.metrics import compute_metrics


def compute_sample_metrics(Depth_computed, Depth_GT):
    metrics_list = []
    for i in range(Depth_computed.shape[0]):
        pred, gt = Depth_computed.squeeze(1).cpu().numpy(), Depth_GT.squeeze(1).cpu().numpy()
        metrics = compute_metrics(pred, gt)
        metrics_list.append(metrics[0])
    return metrics_list

def compute_average_metrics(all_metrics):
    avg_metrics = np.mean(all_metrics, axis=0)
    return {
        'MSE': avg_metrics[0], 'RMS': avg_metrics[1], 'logRMS': avg_metrics[2],
        'AbsRel': avg_metrics[3], 'SqRel': avg_metrics[4], 'Acc_1.25': avg_metrics[5],
        'Acc_1.25^2': avg_metrics[6], 'Acc_1.25^3': avg_metrics[7], 'BadPix': avg_metrics[8],
        'Bumpiness': avg_metrics[9], 'MAE': avg_metrics[10], 'Corr': avg_metrics[11]
    }


def save_metrics_to_csv(all_metrics, results_dir, experiment_name):
    columns = ['Sample', 'MSE', 'RMS', 'logRMS', 'AbsRel', 'SqRel',
               'Acc_1.25', 'Acc_1.25^2', 'Acc_1.25^3', 'BadPix', 'Bumpiness', 'MAE', 'Corr']

    metrics_df = pd.DataFrame(columns=columns)

    for idx, metrics in enumerate(all_metrics):
        new_row = pd.DataFrame([{
            'Sample': f'Sample_{idx}', 'MSE': metrics[0], 'RMS': metrics[1], 'logRMS': metrics[2],
            'AbsRel': metrics[3], 'SqRel': metrics[4], 'Acc_1.25': metrics[5],
            'Acc_1.25^2': metrics[6], 'Acc_1.25^3': metrics[7], 'BadPix': metrics[8],
            'Bumpiness': metrics[9], 'MAE': metrics[10], 'Corr': metrics[11]
        }])

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    avg_metrics = compute_average_metrics(all_metrics)
    avg_row = pd.DataFrame([{'Sample': 'Average', **avg_metrics}])
    metrics_df = pd.concat([metrics_df, avg_row], ignore_index=True)

    metrics_df.to_csv(f'{results_dir}/metrics_{experiment_name}.csv', index=False)

    # Log the average metrics
    avg_metrics_str = ", ".join([f"{key}: {value:.4f}" for key, value in avg_metrics.items()])
    logging.info(f"\nAverage Test Metrics:\n{avg_metrics_str}")
