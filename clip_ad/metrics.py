"""Binary evaluation metrics independent of PyTorch and model inference."""

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from scipy import ndimage


def binary_metrics(truth, scores):
    """AUROC, AP and best positive-class F1 across score thresholds.

    F1-max uses evaluation labels to choose a threshold; it is not F1 at a
    deployment threshold chosen on independent validation data.
    """
    truth, scores = np.asarray(truth).ravel(), np.asarray(scores).ravel()
    if truth.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Labels and finite scores must have matching shapes')
    if np.unique(truth).size < 2:
        return dict(auroc=None, ap=None, f1_max=None)
    precision, recall, _ = precision_recall_curve(truth, scores)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    return dict(auroc=float(roc_auc_score(truth, scores)),
                ap=float(average_precision_score(truth, scores)),
                f1_max=float(f1.max()))


def pixel_aupro(masks, maps, max_steps=200, fpr_limit=0.3):
    """DAPO-compatible sampled AUPRO, with connected regions weighted equally.

    Reproduces DAPO.metrics.cal_pro_score: full-connectivity components, scores
    strictly above each threshold, FPR strictly below the limit, and rescaling
    the retained FPR range to [0, 1]. Sorted scores avoid relabeling masks and
    scanning all pixels at each threshold. Undefined sampled curves return None.
    This follows DAPO's sampled-range normalization, not interpolation at 0.3.
    """
    masks, maps = np.asarray(masks), np.asarray(maps)
    if masks.ndim != 3 or masks.shape != maps.shape or not np.isfinite(maps).all():
        raise ValueError('Masks and finite maps must have matching [images, height, width] shapes')
    if max_steps < 2 or not 0 < fpr_limit <= 1:
        raise ValueError('Need at least two thresholds and an FPR limit in (0, 1]')
    masks = masks.astype(bool)
    if not masks.any() or masks.all():
        return None
    lo, hi = maps.min(), maps.max()
    if lo == hi:
        return None
    thresholds = np.arange(lo, hi, (hi - lo) / max_steps)
    negative_scores = np.sort(maps[~masks])
    fprs = 1 - np.searchsorted(negative_scores, thresholds, side='right') / negative_scores.size
    retained = fprs < fpr_limit
    fprs = fprs[retained]
    if fprs.size < 2 or fprs.max() == fprs.min():
        return None
    thresholds = thresholds[retained]
    pros = np.zeros(thresholds.size, dtype=np.float64)
    count = 0
    for mask, scores in zip(masks, maps):
        # A full 3x3 structure matches skimage.label's default 8-connectivity.
        # Avoid importing skimage: older server wheels can have a NumPy ABI mismatch.
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        for label_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
            if bounds is None:
                continue
            region_scores = np.sort(scores[bounds][labels[bounds] == label_id])
            pros += 1 - np.searchsorted(region_scores, thresholds, side='right') / region_scores.size
            count += 1
    pros /= count
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    return float(auc(fprs, pros))


def summarize_records(records):
    """Compute per-product and macro metrics, excluding undefined categories."""
    if not records:
        raise ValueError('Evaluation dataset is empty')
    metrics = {}
    for name, record in records.items():
        pixels = binary_metrics(record['masks'], record['maps'])
        pixels['aupro'] = pixel_aupro(record['masks'], record['maps'])
        metrics[name] = dict(image=binary_metrics(record['labels'], record['scores']), pixel=pixels)
    macro = {}
    for level in ('image', 'pixel'):
        macro[level] = {}
        for metric in next(iter(metrics.values()))[level]:
            values = [result[level][metric] for result in metrics.values() if result[level][metric] is not None]
            macro[level][metric] = float(np.mean(values)) if values else None
    return metrics, macro
