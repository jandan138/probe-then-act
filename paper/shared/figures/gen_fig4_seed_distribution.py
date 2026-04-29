"""Figure 4: Paired-seed slope chart, M1 vs M7 across all 5 splits.

One small panel per split. Each panel shows two columns (M1, M7) with one
thin line per training seed connecting matched-seed transfer efficiencies.
This directly supports the '2 of 3 matched seeds' statement and exposes
seed-level dispersion that group-bar charts hide.
"""
import sys
sys.path.insert(0, 'figures')
from paper_plot_style import *
import pandas as pd
import numpy as np

per_seed = pd.read_csv('results/ood_eval_per_seed.csv')
splits = SPLIT_ORDER

methods = ['m1_reactive', 'm7_pta']
method_display = ['M1', 'M7']

fig, axes = plt.subplots(1, len(splits), figsize=(9, 2.4), sharey=True)

seed_palette = {42: '#1F77B4', 0: '#FF7F0E', 1: '#2CA02C'}

for ax, split in zip(axes, splits):
    seed_set = sorted(per_seed[per_seed['split'] == split]['seed'].unique())
    means = {}
    for i, m in enumerate(methods):
        for s in seed_set:
            row = per_seed[(per_seed['method'] == m) &
                           (per_seed['split'] == split) &
                           (per_seed['seed'] == s)]
            if len(row):
                means[(m, s)] = row['mean_transfer'].values[0] * 100

    # Connect lines per seed
    for s in seed_set:
        if (methods[0], s) not in means or (methods[1], s) not in means:
            continue
        y0 = means[(methods[0], s)]
        y1 = means[(methods[1], s)]
        c = seed_palette.get(s, '#999')
        slope_color = '#1A8754' if y1 > y0 else '#C44E52'
        ax.plot([0, 1], [y0, y1], color=slope_color, linewidth=1.4,
                alpha=0.6, zorder=2)
        ax.scatter([0, 1], [y0, y1], s=34, color=c, edgecolor='black',
                   linewidth=0.5, zorder=3)

    # Seed-mean line; kept moderate so it does not dominate the per-seed
    # slopes that are the actual point of this plot.
    mean0 = np.mean([means[(methods[0], s)] for s in seed_set
                     if (methods[0], s) in means])
    mean1 = np.mean([means[(methods[1], s)] for s in seed_set
                     if (methods[1], s) in means])
    ax.plot([0, 1], [mean0, mean1], color='black', linewidth=1.4,
            linestyle='-', alpha=0.8, zorder=4)
    ax.scatter([0, 1], [mean0, mean1], s=44, color='black', zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(method_display, fontsize=9)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 100)
    ax.set_title(SPLIT_LABELS[split], fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)

axes[0].set_ylabel('Transfer efficiency (\\%)', fontsize=9.5)

# Highlight elastoplastic with a coloured frame
ep_idx = splits.index('ood_elastoplastic')
for spine in axes[ep_idx].spines.values():
    spine.set_edgecolor('#AA3377')
    spine.set_linewidth(1.5)

# Legend bottom
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='#1A8754', lw=1.4, label='Per-seed slope: M7 > M1'),
    Line2D([0], [0], color='#C44E52', lw=1.4, label='Per-seed slope: M7 < M1'),
    Line2D([0], [0], color='black', lw=1.8, label='Seed-mean'),
]
fig.legend(handles=legend_handles, loc='lower center',
           bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8.5)

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_fig(fig, 'fig4_seed_distribution')
plt.close()
