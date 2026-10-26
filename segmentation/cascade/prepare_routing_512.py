"""Audit native-512 fits and export fold-selected development routing.

Run from the repository root; takes --project-root. Writes <root>/results/routing_cached512_v1 (one OOF routing table per Stage-2 model plus audit.json)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'classification/training'))
from operational_metrics import select_threshold  # noqa: E402

from segmentation import paths  # noqa: E402
from segmentation.cascade.infer_b0_oof import native_backend, validate_checkpoint  # noqa: E402
from segmentation.data import sha256  # noqa: E402


def main():
    data = paths.STAGE2_RESULTS / 'training_patches/ps512_overlap50'
    folds = paths.STAGE2_RESULTS / 'grouped_folds_v1'
    runs = paths.STAGE2_RESULTS / 'stage2_cached512_v1'      # Stage-2 classifier runs on 512 px patches
    seg = paths.RESULTS / 'stage3_native512_v1'              # native 512 px SegFormer-B0 checkpoints
    out = paths.RESULTS / 'routing_cached512_v1'
    index = pd.read_parquet(data / 'index_ps512_overlap50.parquet').merge(
        pd.read_csv(folds / 'scene_folds.csv')[['scene_id', 'fold']],
        on='scene_id', validate='many_to_one')
    hashes = dict(fold_manifest_sha256=sha256(folds / 'scene_folds.csv'),
                  scene_manifest_sha256=sha256(data / 'scene_manifest.csv'),
                  patch_index_sha256=sha256(data / 'index_ps512_overlap50.parquet'))
    records, outputs = [], {}
    for fold in range(1, 6):
        directory = seg / f'segformer_b0_p512_fold{fold:02d}_seed42'
        config = json.loads((directory / 'config.json').read_text())
        history = json.loads((directory / 'history.json').read_text())
        assert len(history) == 50
        best = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        validate_checkpoint(best, fold, 42, hashes, 512)
        assert best['identity'] == config
        model = native_backend(512).model_from_pretrained(best['model_config'])
        model.load_state_dict(best['model_state'], strict=True)
        last = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        assert last['epoch'] == 50 and last['identity'] == config
        records.append(dict(kind='segmentation', fold=fold, batch_size=config['batch_size'],
                            best_epoch=best['epoch'], checkpoint_sha256=sha256(directory / 'best.pt')))
        del model, last, best
    for model_name in ('effnet_b2', 'convnext_tiny'):
        frames = []
        for fold in range(1, 6):
            directory = runs / f'{model_name}_p512_fold{fold:02d}_seed42'
            contract = json.loads((directory / 'config.json').read_text())
            config = contract['config']
            metrics = json.loads((directory / 'metrics.json').read_text())
            history = json.loads((directory / 'history.json').read_text())
            assert (config['model'], config['patch_size'], config['fold'], config['seed']) == (model_name, 512, fold, 42)
            assert config['cache_ram'] and not config['smoke']
            assert metrics['status'] == 'complete' and metrics['completed_epochs'] == 50 and len(history) == 50
            assert contract['source_hashes']['scene_folds.csv'] == hashes['fold_manifest_sha256']
            for row in history:
                assert np.isfinite([row['train_loss'], row['validation_loss'], row['selected']['f2']]).all()
            state = torch.load(directory / 'best_operational.pt', map_location='cpu', weights_only=False)
            assert state['contract'] == contract and state['epoch'] == metrics['best_epoch']
            pred = pd.read_parquet(directory / 'val_predictions.parquet')
            pd.testing.assert_frame_equal(pred, pd.read_parquet(directory / f"validation_epoch_{state['epoch']:03d}.parquet"))
            expected = index[index.fold == fold].set_index('patch_id')
            assert not pred.patch_id.duplicated().any() and set(pred.patch_id) == set(expected.index)
            expected = expected.loc[pred.patch_id]
            assert (pred.scene_id.to_numpy() == expected.scene_id.to_numpy()).all()
            assert set(pred.fold) == {fold} and set(pred.patch_size) == {512}
            for name in ('oil', 'look_alike', 'land', 'sea', 'ship'):
                np.testing.assert_array_equal(pred[f'true_{name}'], expected[f'label_{name}'])
                assert np.isfinite(pred[f'prob_{name}']).all() and pred[f'prob_{name}'].between(0, 1).all()
            selected = select_threshold(pred.true_oil, pred.prob_oil, .9)
            np.testing.assert_allclose(selected['threshold'], metrics['best']['selected']['threshold'])
            np.testing.assert_array_equal(pred.pred_oil, pred.prob_oil >= selected['threshold'])
            r95 = select_threshold(pred.true_oil, pred.prob_oil, .95)
            np.testing.assert_allclose(r95['threshold'], metrics['best']['desired_target']['threshold'])
            pred['pred_oil_r95'] = (pred.prob_oil >= r95['threshold']).astype(int)
            pred['r95_threshold'] = r95['threshold']
            pred['configuration'] = f'{model_name}_p512'
            for column in ('x_orig', 'y_orig', 'valid_pixels'):
                pred[column] = expected[column].to_numpy()
            frames.append(pred)
            records.append(dict(kind=model_name, fold=fold, best_epoch=state['epoch'],
                                selected=selected, r95=r95,
                                predictions_sha256=sha256(directory / 'val_predictions.parquet'),
                                checkpoint_sha256=sha256(directory / 'best_operational.pt')))
            del state
        outputs[f'oof_{model_name}_p512.parquet'] = pd.concat(frames, ignore_index=True)
    out.mkdir(exist_ok=False)
    for filename, frame in outputs.items():
        frame.to_parquet(out / filename, index=False)
    (out / 'audit.json').write_text(json.dumps(dict(records=records, hashes=hashes,
        source_sha256=sha256(__file__), scope='development OOF; thresholds selected on validation; classifier runtime not measured'), indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
