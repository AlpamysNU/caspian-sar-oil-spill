"""Exploratory pixel-retention threshold feasibility on development folds.

Thresholds are selected and summarized on the SAME validation folds. They
describe routing cost, not independently validated operating performance.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths  # noqa: E402
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gate_coverage import get_expected_manifest, load_mask, max_score_map

ANALYSIS = paths.STAGE2_RESULTS/'controlled_analysis_2026-09-13'
OUTPUT = paths.STAGE2_RESULTS/'gate_coverage_2026-09-13'
KEYS = [f'{m}_p{s}' for m in ['effnet_b2','convnext_tiny'] for s in [256,512]]
STRUCTURE = np.ones((3,3),dtype=np.uint8)


def main():
    manifest=get_expected_manifest()
    frames={k:pd.read_parquet(ANALYSIS/f'oof_{k}.parquet') for k in KEYS}
    grouped={k:{scene:g for scene,g in df.groupby('scene_id')} for k,df in frames.items()}
    hist={(k,f):np.zeros(10000,dtype=np.int64) for k in KEYS for f in range(1,6)}
    components=[];scenes=[]
    for i,scene_id in enumerate(manifest.index,1):
        source=manifest.loc[scene_id]
        oil=load_mask(Path(source.mask_path))==1
        labels,n=ndimage.label(oil,structure=STRUCTURE)
        objects=ndimage.find_objects(labels)
        for key in KEYS:
            df=grouped[key][scene_id]
            size=int(df.patch_size.iloc[0]); fold=int(df.fold.iloc[0])
            y=df.y_orig.to_numpy(dtype=int);x=df.x_orig.to_numpy(dtype=int)
            boxes=np.stack([np.clip(y,0,oil.shape[0]),np.clip(y+size,0,oil.shape[0]),
                            np.clip(x,0,oil.shape[1]),np.clip(x+size,0,oil.shape[1])],axis=1)
            scores=max_score_map(df,boxes,oil.shape)
            vals=scores[oil]
            hist[key,fold]+=np.histogram(vals,bins=10000,range=(0,1))[0]
            if len(vals):
                scenes.append(dict(configuration=key,fold=fold,scene_id=scene_id,max_oil_score=float(vals.max())))
            for component_id,sl in enumerate(objects,1):
                if sl is None:continue
                v=scores[sl][labels[sl]==component_id]
                rank=int(np.floor(.1*len(v)))
                components.append(dict(configuration=key,fold=fold,scene_id=scene_id,component_id=component_id,
                                       area_px=len(v),score_for_any=float(v.max()),
                                       score_for_90pct=float(np.partition(v,rank)[rank]),
                                       score_for_all=float(v.min())))
        if i%200==0:print(f'Scored {i}/{len(manifest)} train scenes',flush=True)
    comp=pd.DataFrame(components);scene=pd.DataFrame(scenes)
    comp.to_csv(OUTPUT/'pixel99_component_score_limits.csv',index=False)
    scene.to_csv(OUTPUT/'pixel99_scene_score_limits.csv',index=False)
    rows=[]
    for key in KEYS:
        for fold in range(1,6):
            counts=hist[key,fold]
            total=int(counts.sum())
            kept=counts[::-1].cumsum()[::-1]
            eligible=np.flatnonzero(kept/total>=.99)
            assert len(eligible)
            t=float(eligible[-1]/10000)
            df=frames[key].loc[frames[key].fold==fold]
            routed=int((df.prob_oil>=t).sum())
            size=int(df.patch_size.iloc[0])
            c=comp[(comp.configuration==key)&(comp.fold==fold)]
            s=scene[(scene.configuration==key)&(scene.fold==fold)]
            rows.append(dict(configuration=key,fold=fold,exploratory_threshold=t,oil_pixels=total,
                             retained_oil_pixels=int(kept[eligible[-1]]),oil_retention=float(kept[eligible[-1]]/total),
                             routed_patches=routed,all_patches=len(df),routed_patch_fraction=routed/len(df),
                             repeated_padded_area_px=routed*size*size,
                             components=len(c),components_fully_lost=int((c.score_for_any<t).sum()),
                             components_at_least_90pct=int((c.score_for_90pct>=t).sum()),
                             scenes_with_oil=len(s),scenes_fully_lost=int((s.max_oil_score<t).sum())))
    result=pd.DataFrame(rows)
    result.to_csv(OUTPUT/'exploratory_fold_pixel99_feasibility.csv',index=False)
    summary=[]
    for key,g in result.groupby('configuration'):
        summary.append(dict(configuration=key,oil_retention=g.retained_oil_pixels.sum()/g.oil_pixels.sum(),
                            min_fold_oil_retention=g.oil_retention.min(),
                            routed_patches=int(g.routed_patches.sum()),
                            routed_patch_fraction=g.routed_patches.sum()/g.all_patches.sum(),
                            repeated_padded_area_px=int(g.repeated_padded_area_px.sum()),
                            components_fully_lost=int(g.components_fully_lost.sum()),
                            scenes_fully_lost=int(g.scenes_fully_lost.sum()),
                            thresholds=','.join(f'{x:.4f}' for x in g.sort_values('fold').exploratory_threshold)))
    summary=pd.DataFrame(summary).sort_values('repeated_padded_area_px')
    summary.to_csv(OUTPUT/'exploratory_pixel99_summary.csv',index=False)
    # Preserve the existing fold-r95 decision and only lower a threshold when
    # its Oil pixel coverage is below 99%. Lowering cannot create new misses.
    baseline=pd.read_csv(OUTPUT/'per_scene_coverage.csv')
    run_info=pd.read_csv(ANALYSIS/'per_run.csv')
    conservative=[]
    for row in result.itertuples():
        b=baseline[(baseline.configuration==row.configuration)&(baseline.fold==row.fold)&
                   (baseline.policy=='fold_r95')]
        previous_t=float(run_info[(run_info.configuration==row.configuration)&
                                  (run_info.fold==row.fold)].threshold_r95.iloc[0])
        previous_retention=b.oil_pixels_kept.sum()/b.oil_pixels.sum()
        t=min(previous_t,row.exploratory_threshold) if previous_retention<.99 else previous_t
        changed=t<previous_t
        retained=int(row.retained_oil_pixels) if changed else int(b.oil_pixels_kept.sum())
        d=frames[row.configuration]
        d=d[d.fold==row.fold]
        routed=int((d.prob_oil>=t).sum())
        c=comp[(comp.configuration==row.configuration)&(comp.fold==row.fold)]
        s=scene[(scene.configuration==row.configuration)&(scene.fold==row.fold)]
        conservative.append(dict(configuration=row.configuration,fold=row.fold,
            original_r95_threshold=previous_t,conservative_threshold=t,threshold_lowered=changed,
            oil_pixels=int(b.oil_pixels.sum()),oil_pixels_kept=retained,
            oil_retention=retained/b.oil_pixels.sum(),routed_patches=routed,all_patches=len(d),
            routed_patch_fraction=routed/len(d),repeated_padded_area_px=routed*int(d.patch_size.iloc[0])**2,
            components_fully_lost=int((c.score_for_any<t).sum()),
            scenes_fully_lost=int((s.max_oil_score<t).sum())))
    conservative=pd.DataFrame(conservative)
    assert (conservative.oil_retention >= .99-1e-8).all()
    conservative.to_csv(OUTPUT/'exploratory_conservative_pixel99_by_fold.csv',index=False)
    conservative_summary=[]
    for key,g in conservative.groupby('configuration'):
        conservative_summary.append(dict(configuration=key,folds_threshold_lowered=int(g.threshold_lowered.sum()),
            oil_retention=g.oil_pixels_kept.sum()/g.oil_pixels.sum(),worst_fold_oil_retention=g.oil_retention.min(),
            routed_patches=int(g.routed_patches.sum()),routed_patch_fraction=g.routed_patches.sum()/g.all_patches.sum(),
            repeated_padded_area_px=int(g.repeated_padded_area_px.sum()),
            components_fully_lost=int(g.components_fully_lost.sum()),
            scenes_fully_lost=int(g.scenes_fully_lost.sum())))
    conservative_summary=pd.DataFrame(conservative_summary)
    conservative_summary.to_csv(OUTPUT/'exploratory_conservative_pixel99_summary.csv',index=False)
    (OUTPUT/'pixel99_audit.json').write_text(json.dumps(dict(scope='same 1002 official-train OOF development scenes',
        threshold_grid=0.0001,rule='Maximum score threshold in each validation fold retaining at least 99% of its Oil pixels',
        component_definition='8-connected Oil components; no component missing from threshold calculation',
        conservative_policy='Keep fold-selected 95%-patch-recall thresholds. Only lower those failing 99% Oil-pixel retention, using a 0.0001 score grid. No threshold is raised; previously routed windows remain routed.',
        important_limit='Thresholds selected and evaluated on the same fold using pixel-level ground truth. Results are exploratory feasibility, not held-out validation or official test. A nested calibration split or locked independent test is needed to verify the 99% constraint.'),indent=2)+'\n')
    print(summary.to_string(index=False))
    print('Conservative policy:')
    print(conservative_summary.to_string(index=False))


if __name__=='__main__':main()
