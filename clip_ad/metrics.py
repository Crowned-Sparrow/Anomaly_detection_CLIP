"""Binary and multi-type metrics independent of PyTorch and model inference."""

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


def multitype_metrics(truth, scores, labels):
    """One-vs-rest metrics, including normal, matching DAPO's typed evaluator.

    Scores are channel-last [images, height, width, classes]. AP averages every
    supplied class; F1 follows sklearn's macro over labels in truth or argmax
    predictions, as upstream does. Undefined per-class AUROCs are null and
    excluded from the AUROC mean instead of crashing the evaluation.
    """
    truth, scores = np.asarray(truth), np.asarray(scores)
    if scores.shape != (*truth.shape, len(labels)) or not labels:
        raise ValueError('Typed scores must match masks and the supplied class names')
    if not np.issubdtype(truth.dtype, np.integer) or truth.size == 0:
        raise ValueError('Typed masks must contain integer class IDs and be nonempty')
    if truth.min() < 0 or truth.max() >= len(labels):
        raise ValueError('Typed mask IDs must index the supplied class names')
    truth, scores = truth.ravel(), scores.reshape(-1, len(labels))
    predictions = scores.argmax(axis=1)
    per_defect, f1_values = {}, []
    for index, label in enumerate(labels):
        positive, predicted = truth == index, predictions == index
        values = scores[:, index]
        if not np.isfinite(values).all():
            raise ValueError('Typed scores must be finite')
        support, predicted_count = int(positive.sum()), int(predicted.sum())
        denominator = support + predicted_count
        f1 = float(2 * np.count_nonzero(positive & predicted) / denominator) if denominator else 0.
        if denominator:
            f1_values.append(f1)
        per_defect[label] = dict(
            auroc=float(roc_auc_score(positive, values)) if 0 < support < truth.size else None,
            ap=float(average_precision_score(positive, values)) if support else 0.,
            f1=f1, support=support,
        )
    aucs = [value['auroc'] for value in per_defect.values() if value['auroc'] is not None]
    return dict(auroc=float(np.mean(aucs)) if aucs else None,
                ap=float(np.mean([value['ap'] for value in per_defect.values()])),
                f1=float(np.mean(f1_values))), per_defect


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
