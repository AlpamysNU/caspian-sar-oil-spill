"""E2 precursor (2026-10-03): object-level Oil / Look-alike decision on the saved Stage-3 OOF label maps, no training.

Objects = 8-connected components of the predicted Oil-or-Look-alike mask. Because only label maps are saved, the pooled 'logit' of an object is
approximated by its Oil share f = Oil pixels / (Oil + Look-alike pixels). Variants, each applied to whole objects:
  vote_tau   : object becomes Oil if f >= tau else Look-alike (tau = 0.5 is the majority vote; sweep of tau = decision-threshold sweep)
  oracle_cls : object takes the class (Oil / Look-alike) with more ground-truth pixels inside it; objects without ground-truth Oil/Look-alike
               overlap are left unchanged (ceiling of a perfect Oil-vs-Look-alike object head)
  oracle_full: as oracle_cls, but objects without ground-truth Oil/Look-alike overlap become Sea (ceiling incl. object-level detection)
Also reports the object table (ground-truth majority vs predicted majority, by pixel area) and the distribution of f for flipped objects.
Writes `e2_object_variants.csv`, `e2_objects.csv`, `e2_flipped_f_hist.csv`.
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from segmentation.evaluation.common import CONN8, OIL, read_prediction
from segmentation import paths

LA, SEA = 2, 0
ARMS = {'segformer_b0_shift': 'oof_stage3_b0_shift_256', 'segformer_b2_shift': 'oof_stage3_b2_shift_256',
        'segformer_b2_shift_s43': 'oof_stage3_b2_shift_seed43_256'}
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def counts(v, gt):
    out = []
    for cls in (OIL, LA):
        p, g = v == cls, gt == cls
        out += [int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())]
    return out


def analyse(task, results_root, labels_root):
    scene, fold = task
    gt = np.asarray(Image.open(Path(labels_root) / f'{scene.removeprefix("train_")}.png'))
    rows, objs = [], []
    for arm, run in ARMS.items():
        pred = read_prediction(Path(results_root) / run, fold, scene)
        lab, n = ndimage.label((pred == OIL) | (pred == LA), structure=CONN8)
        rows.append((arm, 'none', counts(pred, gt)))
        if n == 0:
            for name in [f'vote_{t}' for t in TAUS] + ['oracle_cls', 'oracle_full']:
                rows.append((arm, name, counts(pred, gt)))
            continue
        po = np.bincount(lab.ravel(), weights=(pred == OIL).ravel(), minlength=n + 1)
        pl = np.bincount(lab.ravel(), weights=(pred == LA).ravel(), minlength=n + 1)
        go = np.bincount(lab.ravel(), weights=(gt == OIL).ravel(), minlength=n + 1)
        gl = np.bincount(lab.ravel(), weights=(gt == LA).ravel(), minlength=n + 1)
        f = po / np.maximum(po + pl, 1)
        for t in TAUS:
            cls = np.where(f >= t, OIL, LA)
            cls[0] = SEA
            v = np.where(lab > 0, cls[lab], pred)
            rows.append((arm, f'vote_{t}', counts(v, gt)))
        has_gt = (go + gl) > 0
        cls = np.where(go >= gl, OIL, LA)
        rows.append((arm, 'oracle_cls', counts(np.where((lab > 0) & has_gt[lab], cls[lab], pred), gt)))
        rows.append((arm, 'oracle_full', counts(np.where(lab > 0, np.where(has_gt, cls, SEA)[lab], pred), gt)))
        for i in range(1, n + 1):
            objs.append((arm, scene, int(po[i] + pl[i]), float(f[i]), int(go[i]), int(gl[i])))
    return [(a, n, c, scene) for a, n, c in rows], objs


def metrics(c):
    ot, of_, on, lt, lf, ln = c
    return {'oil_recall': ot / (ot + on), 'oil_precision': ot / (ot + of_), 'oil_iou': ot / (ot + of_ + on),
            'la_recall': lt / (lt + ln), 'la_precision': lt / (lt + lf), 'la_iou': lt / (lt + lf + ln)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(ap)
    ap.add_argument('--results-root', type=Path, default=paths.RESULTS)
    ap.add_argument('--labels-root', type=Path, default=paths.DATASET / 'train/labels_1D')
    ap.add_argument('--output', type=Path, default=paths.RESULTS / 'error_analysis_b2_seed_replicate/e2_object_decision')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    base = args.results_root / ARMS['segformer_b2_shift']
    tasks = [(p.name.removesuffix('_prediction.npy'), fold) for fold in range(1, 6)
             for p in sorted((base / f'fold{fold:02d}').glob('*_prediction.npy'))]
    total, objs, per_scene = {}, [], []
    with Pool(args.workers) as pool:
        for r, o in pool.imap_unordered(partial(analyse, results_root=args.results_root, labels_root=args.labels_root),
                                        tasks, chunksize=4):
            for arm, name, c, scene in r:
                if name in ('none', 'vote_0.5', 'oracle_cls', 'oracle_full'):
                    per_scene.append((scene, arm, name, *c))
                total[(arm, name)] = np.add(total.get((arm, name), 0), c)
            objs += o
    args.output.mkdir(parents=True, exist_ok=True)
    per_scene_df = pd.DataFrame(per_scene, columns=['scene_id', 'arm', 'variant', 'oil_tp', 'oil_fp', 'oil_fn', 'la_tp', 'la_fp', 'la_fn'])
    var = pd.DataFrame([{'arm': a, 'variant': n, **metrics(c)} for (a, n), c in total.items()])
    var.to_csv(args.output / 'e2_object_variants.csv', index=False)
    per_scene_df.to_csv(args.output / 'e2_per_scene_counts.csv', index=False)
    ob = pd.DataFrame(objs, columns=['arm', 'scene_id', 'area', 'oil_share', 'gt_oil_px', 'gt_la_px'])
    ob['gt_class'] = np.select([ob.gt_oil_px + ob.gt_la_px == 0, ob.gt_oil_px >= ob.gt_la_px], ['no_gt', 'oil'], 'look_alike')
    ob['pred_class'] = np.where(ob.oil_share >= .5, 'oil', 'look_alike')
    ob.to_csv(args.output / 'e2_objects.csv', index=False)
    tab = ob.groupby(['arm', 'gt_class', 'pred_class']).agg(objects=('area', 'size'), area=('area', 'sum')).reset_index()
    flipped = ob[(ob.gt_class != 'no_gt') & (ob.gt_class != ob.pred_class)]
    hist = flipped.assign(bin=pd.cut(flipped.oil_share, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)) \
        .groupby(['arm', 'gt_class', 'bin'], observed=True).agg(objects=('area', 'size'), area=('area', 'sum')).reset_index()
    hist.to_csv(args.output / 'e2_flipped_f_hist.csv', index=False)
    pd.set_option('display.width', 220)
    print(var[var.arm == 'segformer_b2_shift'].round(4).to_string(index=False))
    for arm in ('segformer_b2_shift', 'segformer_b0_shift'):
        print('\n', arm, 'best Oil IoU variant:', var[var.arm == arm].sort_values('oil_iou').iloc[-1].to_dict())
    print(tab[tab.arm == 'segformer_b2_shift'].to_string(index=False))
    print(hist[hist.arm == 'segformer_b2_shift'].to_string(index=False))
    print(len(tasks), 'scenes')


if __name__ == '__main__':
    main()
