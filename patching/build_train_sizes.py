#!/usr/bin/env python3
"""Build and exhaustively verify the three official-train 50% overlap datasets."""
import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--limit-per-split', type=int)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    for size in (256,384,512):
        output = args.output_root / f'ps{size}_overlap50'
        command = [sys.executable,str(scripts/'build_training_patches.py'),
                   '--dataset-root',str(args.dataset_root),'--output',str(output),
                   '--splits','train','--patch-size',str(size),'--stride',str(size//2)]
        if args.limit_per_split is not None:
            command += ['--limit-per-split',str(args.limit_per_split)]
        subprocess.run(command,check=True)
        subprocess.run([sys.executable,str(scripts/'verify_training_patches.py'),
                        '--output',str(output)],check=True)
