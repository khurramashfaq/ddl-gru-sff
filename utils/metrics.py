import numpy as np
import skimage.filters as skf
from scipy.stats import pearsonr



def compute_metrics(model_computed, groundtruth, mse_factor=1, accthrs=[1.25, 1.25**2, 1.25**3], bumpiness_clip=0.05, ignore_zero=True):
    '''
    Expects depthmap in numpy with dims : B,H,W (There is no channel needed)
    '''
    metrics = np.zeros((1, 9 + len(accthrs)), dtype=float)
    batch_size = model_computed.shape[0]

    for b in range(batch_size):
        pred_ = np.copy(model_computed[b])
        target_ = np.copy(groundtruth[b])

        if target_.sum() == 0:
            continue

        if ignore_zero:
            pred_[target_ == 0.0] = 0.0
            num_pixels = (target_ > 0.0).sum()  # Number of valid pixels
        else:
            num_pixels = target_.size


        # Mean Squared Error (MSE)
        mse = np.square(pred_ - target_).sum() / num_pixels * mse_factor
        metrics[0, 0] += mse

        # Root Mean Squared Error (RMS)
        rms = np.sqrt(mse)
        metrics[0, 1] += rms

        # Log RMS
        log_rms = np.ma.log(pred_) - np.ma.log(target_)
        log_rms_value = np.sqrt(np.square(log_rms).sum() / num_pixels)
        metrics[0, 2] += log_rms_value

        # Absolute Relative Error (AbsRel)
        abs_rel = np.ma.divide(np.abs(pred_ - target_), target_).sum() / num_pixels
        metrics[0, 3] += abs_rel

        # Square Relative Error (SqRel)
        sq_rel = np.ma.divide(np.square(pred_ - target_), target_).sum() / num_pixels
        metrics[0, 4] += sq_rel

        # Accuracy under thresholds
        acc = np.ma.maximum(np.ma.divide(pred_, target_), np.ma.divide(target_, pred_))
        for i, thr in enumerate(accthrs):
            metrics[0, 5 + i] += (acc < thr).sum() / num_pixels * 100.

        # Bad pixel percentage (BadPix)
        bad_pix = (np.abs(pred_ - target_) > 0.07).sum() / num_pixels * 100.
        metrics[0, 8] += bad_pix

        # Bumpiness metric
        diff = np.asarray(pred_ - target_, dtype='float64')
        chn = diff.shape[2] if len(diff.shape) > 2 else 1
        bumpiness = np.zeros_like(pred_).astype('float')

        for c in range(chn):
            diff_2d = diff[:, :, c] if chn > 1 else diff
            dx = skf.scharr_v(diff_2d)
            dy = skf.scharr_h(diff_2d)
            dxx = skf.scharr_v(dx)
            dxy = skf.scharr_h(dx)
            dyy = skf.scharr_h(dy)
            dyx = skf.scharr_v(dy)

            hessian_norm = np.sqrt(np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
            bumpiness += np.clip(hessian_norm, 0, bumpiness_clip)

        bumpiness_sum = bumpiness[target_ > 0].sum() if ignore_zero else bumpiness.sum()
        metrics[0, 9] += bumpiness_sum / num_pixels * 100.

        # Mean Absolute Error (MAE)
        valid_mask = target_ > 0 if ignore_zero else np.ones_like(target_, dtype=bool)
        mae = np.mean(np.abs(pred_[valid_mask] - target_[valid_mask]))
        metrics[0, 10] += mae

        # Correlation (Pearson)
        valid_mask = target_ > 0
        if np.any(valid_mask):
            corr = pearsonr(pred_[valid_mask].flatten(), target_[valid_mask].flatten())[0]
        else:
            corr = 0.0
        metrics[0, 11] += corr

    metrics = metrics / batch_size

    return metrics
