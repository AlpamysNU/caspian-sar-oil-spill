"""Evaluate Stage 2 routing against original Oil masks on official train OOF scenes.

Uses no test labels, model inference, or new training. Requires the outputs of
analyze_controlled_runs.py. Reports real union coverage of accepted windows,
not patch-label recall. Thresholds were selected on the same development data.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths  # noqa: E402
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT/'dataset_analysis/scripts'))
from stage2_geometry_audit import load_mask, padding_and_positions  # noqa: E402

MODELS = ('effnet_b2', 'convnext_tiny')
SIZES = (256, 512)
POLICIES = ('fold_r90', 'fold_r95', 'pooled_r95')
STRUCTURE = np.ones((3, 3), dtype=np.uint8)
AREA_BINS = ((1, 8, '1-8'), (9, 24, '9-24'), (25, None, '>=25'))


def get_expected_manifest():
    path = paths.STAGE2_RESULTS/'training_patches/ps256_overlap50/scene_manifest.csv'
    return pd.read_csv(path).set_index('scene_id')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def windows_and_validate(frame, shape, size, oil):
    """Check every index window against the actual padded grid and Oil mask."""
    h, w = shape
    stride = size//2
    pad_top, _, ys = padding_and_positions(h, size, stride)
    pad_left, _, xs = padding_and_positions(w, size, stride)
    expected = {(y-pad_top, x-pad_left) for y in ys for x in xs}
    actual = set(zip(frame.y_orig.astype(int), frame.x_orig.astype(int)))
    assert actual == expected and len(frame) == len(expected)
    coords = frame[['y_orig','x_orig']].to_numpy(dtype=np.int32)
    top = np.clip(coords[:, 0], 0, h)
    left = np.clip(coords[:, 1], 0, w)
    bottom = np.clip(coords[:, 0]+size, 0, h)
    right = np.clip(coords[:, 1]+size, 0, w)
    assert np.all(bottom > top) and np.all(right > left)
    counts = integral_counts(oil, top, bottom, left, right)
    np.testing.assert_array_equal(counts, frame.pixels_oil.to_numpy())
    np.testing.assert_array_equal((bottom-top)*(right-left), frame.valid_pixels.to_numpy())
    np.testing.assert_array_equal(counts > 0, frame.true_oil.to_numpy())
    return np.stack([top,bottom,left,right], axis=1)


def integral_counts(oil, top, bottom, left, right):
    integral = np.pad(oil.astype(np.int32), ((1,0),(1,0))).cumsum(0).cumsum(1)
    return integral[bottom,right]-integral[top,right]-integral[bottom,left]+integral[top,left]


def max_score_map(frame, boxes, shape):
    score_map = np.full(shape, -np.inf, dtype=np.float32)
    scores = frame.prob_oil.to_numpy(dtype=np.float32)
    for score, (t,b,l,r) in zip(scores, boxes):
        np.maximum(score_map[t:b,l:r], score, out=score_map[t:b,l:r])
    assert np.isfinite(score_map).all(), 'Grid leaves pixels uncovered'
    return score_map


def component_counts(labels, n, covered, areas):
    inside = np.bincount(labels[covered], minlength=n+1)[1:]
    assert len(inside) == n
    return inside


def evaluate_scene(scene_id, fold, key, size, frame, oil, labels, areas, boxes, score_map,
                   pooled_t, scene_area, per_scene, per_component, score_hist):
    oil_scores = score_map[oil]
    score_hist[key] += np.histogram(oil_scores, bins=10000, range=(0,1))[0]
    thresholds = dict(fold_r90=float(frame.oil_threshold.iloc[0]),
                      fold_r95=float(frame.loc[0, 'r95_threshold']), pooled_r95=pooled_t)
    for policy in POLICIES:
        threshold = thresholds[policy]
        accept = frame.prob_oil.to_numpy() >= threshold
        if policy == 'fold_r90':
            np.testing.assert_array_equal(accept, frame.pred_oil)
        elif policy == 'fold_r95':
            np.testing.assert_array_equal(accept, frame.pred_oil_r95)
        covered = score_map >= threshold
        oil_kept = int(np.count_nonzero(oil & covered))
        unique_area = int(np.count_nonzero(covered))
        repeated_valid = int(((boxes[:,1]-boxes[:,0])*(boxes[:,3]-boxes[:,2]))[accept].sum())
        routed = int(accept.sum())
        scene_row = dict(configuration=key, size=size, model=key.split('_p')[0], fold=fold,
                         scene_id=scene_id, policy=policy, threshold=threshold,
                         oil_pixels=int(oil.sum()), oil_pixels_kept=oil_kept,
                         oil_pixels_lost=int(oil.sum())-oil_kept,
                         oil_fraction_kept=oil_kept/max(int(oil.sum()),1),
                         scene_area=scene_area, unique_scene_area_covered=unique_area,
                         patches=len(frame), patches_routed=routed,
                         full_padded_area=len(frame)*size*size,
                         routed_padded_area=routed*size*size,
                         all_valid_area=int(((boxes[:,1]-boxes[:,0])*(boxes[:,3]-boxes[:,2])).sum()),
                         routed_valid_area=repeated_valid,
                         oil_components=len(areas))
        inside = component_counts(labels, len(areas), covered, areas)
        scene_row.update(components_fully_lost=int((inside == 0).sum()),
                         components_any_kept=int((inside > 0).sum()),
                         components_at_least_90pct=int((inside >= .9*areas).sum()),
                         components_fully_kept=int((inside == areas).sum()))
        per_scene.append(scene_row)
        for component_id, (area, kept) in enumerate(zip(areas, inside), 1):
            group = next(name for lo,hi,name in AREA_BINS if area >= lo and (hi is None or area <= hi))
            per_component.append(dict(configuration=key, size=size, fold=fold, scene_id=scene_id,
                                      policy=policy, component_id=component_id, area_px=int(area),
                                      area_group=group, pixels_kept=int(kept), fraction_kept=float(kept/area)))


def summarize(scenes, components):
    rows=[]
    for (key,policy), g in scenes.groupby(['configuration','policy'], sort=True):
        c=components[(components.configuration==key)&(components.policy==policy)]
        row=dict(configuration=key, policy=policy, scenes=len(g), oil_scenes=int((g.oil_pixels>0).sum()),
                 oil_pixels=int(g.oil_pixels.sum()), oil_pixels_kept=int(g.oil_pixels_kept.sum()),
                 oil_retention=g.oil_pixels_kept.sum()/g.oil_pixels.sum(),
                 worst_fold_oil_retention=min(x.oil_pixels_kept.sum()/x.oil_pixels.sum() for _,x in g.groupby('fold')),
                 oil_scenes_fully_lost=int(((g.oil_pixels>0)&(g.oil_pixels_kept==0)).sum()),
                 components=len(c), components_fully_lost=int((c.fraction_kept==0).sum()),
                 components_at_least_90pct=int((c.fraction_kept>=.9).sum()),
                 components_fully_kept=int((c.fraction_kept==1).sum()),
                 patches=int(g.patches.sum()), routed_patches=int(g.patches_routed.sum()),
                 routed_patch_fraction=g.patches_routed.sum()/g.patches.sum(),
                 padded_area_fraction=g.routed_padded_area.sum()/g.full_padded_area.sum(),
                 valid_area_fraction=g.routed_valid_area.sum()/g.all_valid_area.sum(),
                 unique_scene_area_fraction=g.unique_scene_area_covered.sum()/g.scene_area.sum(),
                 repeated_padded_area_px=int(g.routed_padded_area.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary, groups, output):
    fig, axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    colors={'effnet_b2_p256':'#0072B2','effnet_b2_p512':'#56B4E9',
            'convnext_tiny_p256':'#D55E00','convnext_tiny_p512':'#CC79A7'}
    for key,g in summary.groupby('configuration'):
        axes[0].plot(g.routed_patch_fraction,g.oil_retention,'o-',label=key,color=colors[key])
        axes[1].plot(g.unique_scene_area_fraction,g.oil_retention,'o-',label=key,color=colors[key])
        for _,r in g.iterrows():
            axes[0].annotate(r.policy.replace('fold_',''),(r.routed_patch_fraction,r.oil_retention),fontsize=7)
    axes[0].set(xlabel='Fraction of windows routed',ylabel='Oil pixel retention')
    axes[1].set(xlabel='Fraction of unique scene area routed',ylabel='Oil pixel retention')
    for ax in axes:
        ax.axhline(.99,color='black',linestyle=':',linewidth=1,label='99% target')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Development folds: Oil coverage of union of routed windows')
    fig.savefig(output/'gate_coverage_tradeoff.png',dpi=180);fig.savefig(output/'gate_coverage_tradeoff.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for key,g in groups.groupby('configuration'):
        g=g[g.policy=='fold_r95'].set_index('area_group').reindex([x[2] for x in AREA_BINS])
        ax.plot(g.index,g.component_any_retained,'o-',label=key,color=colors[key])
    ax.set(xlabel='Oil component area (pixels)',ylabel='Fraction with any pixel routed',ylim=(0,1.02))
    ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title('Component survival at fold-selected 95% patch-recall thresholds')
    fig.savefig(output/'component_survival.png',dpi=180);fig.savefig(output/'component_survival.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-root',type=Path,default=paths.STAGE2_RESULTS/'controlled_analysis_2026-09-13')
    p.add_argument('--output',type=Path,default=paths.STAGE2_RESULTS/'gate_coverage_2026-09-13')
    p.add_argument('--limit',type=int,default=None,help='Smoke check on first N scenes; results marked partial')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=get_expected_manifest()
    assert set(manifest.split)=={'train'} and len(manifest)==1002
    keys=[f'{m}_p{s}' for m in MODELS for s in SIZES]
    frames={key:pd.read_parquet(args.analysis_root/f'oof_{key}.parquet') for key in keys}
    per_run=pd.read_csv(args.analysis_root/'per_run.csv')
    pooled=pd.read_csv(args.analysis_root/'exploratory_pooled_thresholds.csv')
    pooled_t={r.configuration:float(r.exploratory_pooled_threshold) for r in pooled.itertuples() if r.recall_constraint==.95}
    for key,df in frames.items():
        t=per_run[per_run.configuration==key].set_index('fold').threshold_r95.to_dict()
        df['r95_threshold']=df.fold.map(t)
        assert df.r95_threshold.notna().all()
        assert df.patch_id.nunique()==len(df)
    scene_ids=list(manifest.index)
    if args.limit is not None: scene_ids=scene_ids[:args.limit]
    per_scene=[];per_component=[];score_hist={key:np.zeros(10000,dtype=np.int64) for key in keys}
    for i,scene_id in enumerate(scene_ids,1):
        source=manifest.loc[scene_id]
        mask_path=Path(source.mask_path)
        assert sha(mask_path)==source.mask_sha256, scene_id
        mask=load_mask(mask_path)
        assert mask.shape==(source.height,source.width)
        oil=mask==1
        labels,n=ndimage.label(oil,structure=STRUCTURE)
        areas=np.bincount(labels.ravel(),minlength=n+1)[1:]
        assert int(areas.sum())==int(oil.sum())
        for size in SIZES:
            subgroup={m:frames[f'{m}_p{size}'].loc[frames[f'{m}_p{size}'].scene_id==scene_id].reset_index(drop=True) for m in MODELS}
            a,b=subgroup.values()
            assert len(a)==len(b)>0
            pd.testing.assert_frame_equal(a[['patch_id','y_orig','x_orig','pixels_oil','true_oil']].sort_values('patch_id').reset_index(drop=True),
                                          b[['patch_id','y_orig','x_orig','pixels_oil','true_oil']].sort_values('patch_id').reset_index(drop=True))
            for model,frame in subgroup.items():
                key=f'{model}_p{size}'
                boxes=windows_and_validate(frame,mask.shape,size,oil)
                score_map=max_score_map(frame,boxes,mask.shape)
                evaluate_scene(scene_id,int(frame.fold.iloc[0]),key,size,frame,oil,labels,areas,boxes,score_map,
                               pooled_t[key],int(oil.size),per_scene,per_component,score_hist)
        if i%100==0:print(f'Validated {i}/{len(scene_ids)} train scenes',flush=True)
    scenes=pd.DataFrame(per_scene); components=pd.DataFrame(per_component)
    assert len(scenes)==len(scene_ids)*len(keys)*len(POLICIES)
    scenes.to_csv(args.output/'per_scene_coverage.csv',index=False)
    components.to_csv(args.output/'per_component_coverage.csv',index=False)
    summary=summarize(scenes,components);summary.to_csv(args.output/'configuration_coverage.csv',index=False)
    groups=[]
    for (key,policy,area_group),g in components.groupby(['configuration','policy','area_group']):
        groups.append(dict(configuration=key,policy=policy,area_group=area_group,components=len(g),
                           component_any_retained=float((g.fraction_kept>0).mean()),
                           component_at_least_90pct=float((g.fraction_kept>=.9).mean()),
                           component_fully_retained=float((g.fraction_kept==1).mean())))
    groups=pd.DataFrame(groups);groups.to_csv(args.output/'component_area_groups.csv',index=False)
    # Exploratory sweep: maximum probability among windows covering each Oil pixel.
    # Histograms use 1e-4 score bins; threshold values are lower bin edges, approximate.
    sweep=[]
    for key,hist in score_hist.items():
        total=hist.sum()
        for idx in range(0,10001,10):
            t=idx/10000
            kept=int(hist[idx:].sum())
            sweep.append(dict(configuration=key,threshold=t,oil_pixels=int(total),approx_oil_pixels_kept=kept,
                              approx_oil_retention=kept/total))
    sweep=pd.DataFrame(sweep);sweep.to_csv(args.output/'exploratory_pixel_threshold_sweep.csv',index=False)
    plot(summary,groups,args.output)
    meta=dict(scope='official train OOF development scenes only',partial=args.limit is not None,
              scenes=len(scene_ids),models=list(MODELS),sizes=list(SIZES),policies=list(POLICIES),
              mask_integrity='Original RGB PNG SHA-256 checked against frozen train scene manifest, decoded with strict palette',
              geometry_integrity='Every index patch compared with padded grid, original coordinates, valid area and Oil pixel count',
              coverage='Union of accepted original-space rectangles; no repeated credit to overlapping Oil pixels',
              components='8-connected Oil mask regions, separately assessed for any, >=90%, and full pixel coverage',
              thresholds='fold_r90/fold_r95 were selected from each validation fold; pooled_r95 from combined development predictions; all are optimistically in-sample for threshold selection',
              sweep='0.001 threshold grid, approximate 0.0001 probability histogram bins; exploratory; selection requires independent confirmation',
              warning='Accepted-window fractions and padded area are workload proxies, not measured Stage 3 latency or final Oil IoU')
    (args.output/'audit.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(summary[['configuration','policy','oil_retention','worst_fold_oil_retention','components_fully_lost','oil_scenes_fully_lost','routed_patch_fraction','valid_area_fraction']].to_string(index=False))
    print('Output:',args.output)


if __name__=='__main__':main()
