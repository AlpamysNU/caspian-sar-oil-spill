"""Build shared, deterministic scene folds from verified official-train indices."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))   # repository root
from segmentation import paths
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ('oil', 'look_alike', 'land', 'sea', 'ship')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_folds(features, groups, n_folds=5, seed=42):
    """Greedily balance normalized feature totals; never split a group."""
    if len(set(groups)) < n_folds:
        raise ValueError('Fewer independent groups than folds')
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    values = np.array([features[np.asarray(groups) == g].sum(axis=0) for g in unique])
    sizes = np.array([sum(np.asarray(groups) == g) for g in unique])
    totals = values.sum(axis=0)
    normalized = values / np.maximum(totals, 1)
    order = sorted(rng.permutation(len(unique)), key=lambda i: -normalized[i].max())
    counts = np.zeros(n_folds)
    loads = np.zeros((n_folds, values.shape[1]))
    allocation = {}
    target = len(groups) / n_folds
    for i in order:
        scores = []
        for fold in range(n_folds):
            delta = ((loads[fold] + normalized[i]) ** 2 - loads[fold] ** 2).sum()
            delta += ((counts[fold] + sizes[i]) ** 2 - counts[fold] ** 2) / target ** 2
            scores.append(delta)
        fold = int(np.argmin(scores))
        loads[fold] += normalized[i]
        counts[fold] += sizes[i]
        allocation[unique[i]] = fold + 1
    return np.array([allocation[g] for g in groups])


def build(root, output, n_folds=5, seed=42):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    manifests, indices, hashes, blocks = [], {}, {}, []
    for size in (256, 384, 512):
        folder = root / f'ps{size}_overlap50'
        manifest_path = folder / 'scene_manifest.csv'
        index_path = folder / f'index_ps{size}_overlap50.parquet'
        verification_path = folder / 'verification.json'
        verification = json.loads(verification_path.read_text())
        if (verification.get('all_patch_pixels_and_labels_checked') is not True
                or verification.get('splits') != ['train']
                or verification.get('uniform_reconstruction_max_error') != 0):
            raise ValueError(f'Incomplete verification: {folder}')
        manifest = pd.read_csv(manifest_path).sort_values('scene_id').reset_index(drop=True)
        index = pd.read_parquet(index_path)
        if manifest.scene_id.duplicated().any() or index.patch_id.duplicated().any():
            raise ValueError('Duplicate scene or patch identifiers')
        if set(manifest.split) != {'train'} or set(index.split) != {'train'}:
            raise ValueError('Only official train is permitted')
        if set(index.scene_id) != set(manifest.scene_id):
            raise ValueError('Index and source scene sets differ')
        if len(index) != verification['patches'] or len(manifest) != verification['scenes']:
            raise ValueError('Verification counts differ')
        if set(index.patch_size) != {size} or set(index.stride) != {size // 2}:
            raise ValueError('Unexpected patch geometry')
        for c in CLASSES:
            if not index[f'label_{c}'].equals((index[f'pixels_{c}'] > 0).astype(index[f'label_{c}'].dtype)):
                raise ValueError(f'Inconsistent label: {c}')
        indices[size] = index
        manifests.append(manifest)
        grouped = index.groupby('scene_id')
        features = grouped[[f'label_{c}' for c in CLASSES]].sum()
        features['oil_pixel_exposure'] = grouped.pixels_oil.sum()
        features['oil_look_alike'] = index.assign(mixed=index.label_oil * index.label_look_alike).groupby('scene_id').mixed.sum()
        blocks.append(features.reindex(manifest.scene_id).to_numpy())
        for path in (manifest_path, index_path, verification_path):
            hashes[str(path.resolve())] = sha256(path)
    base = manifests[0]
    for manifest in manifests[1:]:
        if not base.equals(manifest):
            raise ValueError('Source manifests differ across patch sizes')
    if base[['group_id', 'image_sha256']].isna().any().any():
        raise ValueError('Missing group or source fingerprint')
    # Union existing source groups and exact image-file duplicates.
    parent = {s: s for s in base.scene_id}
    def find(s):
        while parent[s] != s:
            s = parent[s]
        return s
    for column in ('group_id', 'image_sha256'):
        for _, rows in base.groupby(column):
            first = rows.scene_id.iloc[0]
            for scene in rows.scene_id:
                parent[find(scene)] = find(first)
    groups = np.array([find(s) for s in base.scene_id])
    presence = indices[256].groupby('scene_id')[[f'label_{c}' for c in CLASSES]].max().reindex(base.scene_id)
    features = np.concatenate([presence.to_numpy(), *blocks], axis=1)
    folds = assign_folds(features, groups, n_folds, seed)
    result = base[['scene_id', 'split', 'image_sha256', 'mask_sha256']].copy()
    result['group_id'] = groups
    result['fold'] = folds
    reports = []
    for size, index in indices.items():
        joined = index.merge(result[['scene_id', 'fold']], on='scene_id', validate='many_to_one')
        for fold in range(1, n_folds + 1):
            for role, subset in [('train', joined[joined.fold != fold]), ('validation', joined[joined.fold == fold])]:
                row = dict(patch_size=size, fold=fold, role=role, scenes=subset.scene_id.nunique(), patches=len(subset))
                if not len(subset):
                    raise ValueError('Empty partition')
                for c in CLASSES:
                    positives = int(subset[f'label_{c}'].sum())
                    if positives == 0 or positives == len(subset):
                        raise ValueError(f'Class {c} missing positives or negatives: {size}/{fold}/{role}')
                    row[f'{c}_positive_patches'] = positives
                    row[f'{c}_positive_rate'] = positives / len(subset)
                    row[f'{c}_positive_scenes'] = int(subset.groupby('scene_id')[f'label_{c}'].max().sum())
                row['oil_pixel_exposure'] = int(subset.pixels_oil.sum())
                row['oil_look_alike_patches'] = int((subset.label_oil * subset.label_look_alike).sum())
                reports.append(row)
        for fold in range(1, n_folds + 1):
            train = result[result.fold != fold]
            val = result[result.fold == fold]
            for key in ('scene_id', 'group_id', 'image_sha256'):
                if set(train[key]) & set(val[key]):
                    raise ValueError(f'Partition leakage: {key}')
    output.mkdir(parents=True)
    result.to_csv(output / 'scene_folds.csv', index=False)
    pd.DataFrame(reports).to_csv(output / 'fold_distribution.csv', index=False)
    metadata = dict(version=1, algorithm='normalized-feature greedy group balancing v1', seed=seed,
                    n_folds=n_folds, scenes=len(base), groups=len(set(groups)),
                    fold_scene_counts=result.groupby('fold').size().to_dict(),
                    input_sha256=hashes, script_sha256=sha256(Path(__file__)),
                    manifest_sha256=sha256(output / 'scene_folds.csv'),
                    numpy_version=np.__version__, pandas_version=pd.__version__,
                    checks='All partitions nonempty; all classes have positives and negatives; zero scene/group/image-hash overlap',
                    limitations=['Acquisition/incident independence unavailable.',
                                 'Image-file hashes detect exact byte duplicates only, not near duplicates.',
                                 'Oil pixel exposure is overlap-weighted, not unique scene Oil area.'])
    (output / 'audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ('scenes', 'groups', 'fold_scene_counts', 'manifest_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=paths.STAGE2_RESULTS / 'training_patches')
    parser.add_argument('--output', type=Path, default=paths.STAGE2_RESULTS / 'grouped_folds_v1')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error('--folds must be at least 2')
    build(args.root, args.output, args.folds, args.seed)
