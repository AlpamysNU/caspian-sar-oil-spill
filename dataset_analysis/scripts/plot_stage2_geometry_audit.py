#!/usr/bin/env python3
"""Create a compact visual summary for the Stage-2 geometry audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding='utf-8'))['configurations']
    labels = [key.replace('_', '/') for key in data]
    total = np.array([data[key]['patches_total'] for key in data])
    passed = np.array([data[key]['oil_positive_patches'] for key in data])
    pct = passed / total * 100
    saved = 100 - pct

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_count, ax_gate) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    x = np.arange(len(labels))
    bars_total = ax_count.bar(x, total, color='#B7C9E2', label='All Stage 2 patches')
    bars_pass = ax_count.bar(x, passed, color='#E76F51', label='Passed to Stage 3 (ideal gate)')
    ax_count.set_xticks(x, labels)
    ax_count.set_ylabel('Patch count on official train')
    ax_count.set_title('Workload across patch grids')
    ax_count.legend(loc='upper right')
    for bar, value in zip(bars_total, total):
        ax_count.text(bar.get_x()+bar.get_width()/2, value, f'{value:,}'.replace(',', ' '), ha='center', va='bottom', fontsize=9)
    for bar, value in zip(bars_pass, passed):
        ax_count.text(bar.get_x()+bar.get_width()/2, value, f'{value:,}'.replace(',', ' '), ha='center', va='bottom', fontsize=9, color='#9B3E2C')

    bars = ax_gate.bar(x, saved, color='#2A9D8F')
    ax_gate.set_xticks(x, labels)
    ax_gate.set_ylim(0, 105)
    ax_gate.set_ylabel('Patches potentially skipped by Stage 3, %')
    ax_gate.set_title('Potential Stage 3 workload reduction')
    for bar, pct_pass, pct_saved in zip(bars, pct, saved):
        ax_gate.text(bar.get_x()+bar.get_width()/2, pct_saved/2, f'reduction\n{pct_saved:.1f}%', ha='center', va='center', color='white', weight='bold')
        ax_gate.text(bar.get_x()+bar.get_width()/2, pct_saved+2, f'to Stage 3: {pct_pass:.1f}%', ha='center', fontsize=9)

    fig.suptitle('M4D official train: patch geometry and ideal Oil gate', fontsize=15, weight='bold')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches='tight')
    print(args.output)


if __name__ == '__main__':
    main()
