"""Where things live. Every analysis, scoring and figure script resolves its inputs and outputs from one project root.

The root is taken, in this order, from `--project-root DIR` on the command line, the environment variable
`OILSEG_PROJECT_ROOT`, or the current directory. The option is removed from `sys.argv` when this module is imported. Inside the root the scripts expect

    <root>/oil-spill-detection-dataset/   the unpacked M4D-SAR-OilSpill dataset (train/images, train/labels_1D, ...)
    <root>/results/                       Stage-3 outputs (oof_*, error_analysis_*, label_review/, ...)
    <root>/stage2_results/                Stage-2 outputs (grouped_folds_v1, training_patches, ...)

Individual input and output folders can still be given by their own arguments where a script offers them.
"""
import argparse
import os
import sys
from pathlib import Path


def _root_from_argv():
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--project-root', type=Path)
    known, rest = probe.parse_known_args()
    sys.argv[1:] = rest   # the option is consumed here, so scripts that read positional arguments are not disturbed by it
    chosen = known.project_root or os.environ.get('OILSEG_PROJECT_ROOT') or Path.cwd()
    return Path(chosen).expanduser().resolve()


PROJECT_ROOT = _root_from_argv()
DATASET = PROJECT_ROOT / 'oil-spill-detection-dataset'
RESULTS = PROJECT_ROOT / 'results'
STAGE2_RESULTS = PROJECT_ROOT / 'stage2_results'


def add_root_argument(parser):
    """Declare `--project-root` on a script's own parser so that it shows up in --help (the value is read above)."""
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT,
                        help='Folder holding oil-spill-detection-dataset/, results/ and stage2_results/ '
                             '(default: $OILSEG_PROJECT_ROOT or the current directory)')
    return parser


def resolve(path):
    """Relative output and input folders are taken relative to the project root, absolute ones are kept."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
