"""Exact tied-score PR thresholds and non-interpolated average precision."""
import numpy as np

NAMES = ('oil', 'look_alike', 'land', 'sea', 'ship')


def select_threshold(y, scores, minimum=.90):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError('Expected matching nonempty vectors')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid targets or scores')
    if not 0 < minimum <= 1 or y.sum() in (0, len(y)):
        raise ValueError('PR selection requires both classes and a valid recall bound')
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tp = np.cumsum(y[order])[ends]
    precision, recall = tp / (ends + 1), tp / y.sum()
    f2 = 5 * tp / (4 * y.sum() + ends + 1)
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision))
    eligible = np.flatnonzero(recall >= minimum)
    i = max(eligible, key=lambda j: (f2[j], precision[j], recall[j], sorted_scores[ends[j]]))
    return dict(threshold=float(sorted_scores[ends[i]]), precision=float(precision[i]),
                recall=float(recall[i]), f2=float(f2[i]), ap=ap)


def report(y, probabilities, threshold):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    if y.shape != probabilities.shape or y.ndim != 2 or y.shape[1] != 5:
        raise ValueError('Expected N by 5 arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError('Invalid targets or probabilities')
    pred = probabilities >= np.array([threshold, .5, .5, .5, .5])
    tp, fp, fn = [(condition).sum(0) for condition in
                  (pred & (y == 1), pred & (y == 0), ~pred & (y == 1))]
    precision, recall = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    result = {name: dict(precision=float(precision[i]), recall=float(recall[i]),
                        f1=float(f1[i]), fn=int(fn[i]), fp=int(fp[i]), fnr=float(1-recall[i]))
              for i, name in enumerate(NAMES)}
    prevalence = float(y[:, 0].mean())
    result.update(macro_f1=float(f1.mean()), weighted_f1=float(np.average(f1, weights=y.sum(0))),
                  micro_f1=float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(), 1)),
                  routed_fraction=float(pred[:, 0].mean()),
                  accept_all=dict(recall=1., precision=prevalence, f2=5*prevalence/(4*prevalence+1), routed_fraction=1.))
    return result
