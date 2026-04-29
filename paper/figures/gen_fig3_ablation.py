"""Figure 3: Cleveland dot-plot matrix for ablation study on OOD-elastoplastic.

Replaces the previous 3-bar-panel layout. Each row is a method, each column
is a metric; the central marker is the seed mean and the horizontal whisker
is the seed range. Spill is plotted as 'Retained material (100 - spill)'
so all three columns are higher-is-better.
"""
import sys
sys.path.insert(0, 'figures')
from paper_plot_style import *
import pandas as pd
import numpy as np

df = pd.read_csv('results/main_results.csv')
per_seed = pd.read_csv('results/ood_eval_per_seed.csv')

# Order: Full at top to anchor the comparison, M1 immediately below it,
# then ablations falling away. This mirrors the rhetorical claim in the
# text: "both ablations fall below the reactive baseline".
methods = ['m7_pta', 'm1_reactive', 'm7_nobelief', 'm7_noprobe']
method_display = {
    'm7_pta':      'M7 \\PTA{} Full',
    'm1_reactive': 'M1 Reactive',
    'm7_nobelief': 'M7-NoBelief',
    'm7_noprobe':  'M7-NoProbe',
}
colors = {
    'm7_pta':      COLORS['m7_pta'],
    'm7_nobelief': COLORS['m7_nobelief'],
    'm7_noprobe':  COLORS['m7_noprobe'],
    'm1_reactive': COLORS['m1_reactive'],
}

split = 'ood_elastoplastic'
ep = df[df['split'] == split]
ep_seed = per_seed[per_seed['split'] == split]

panels = [
    ('mean_transfer', None,                 'Transfer efficiency (\\%)'),
    ('mean_spill',    'retained',           'Retained material 100$-$spill (\\%)'),
    ('success_rate',  None,                 'Success rate (\\%)'),
]

fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)

for ax, (key, transform, title) in zip(axes, panels):
    means = []
    seeds = []
    for m in methods:
        row = ep[ep['method'] == m]
        seed_vals = ep_seed[ep_seed['method'] == m][key].values
        if transform == 'retained':
            mean_val = (1.0 - row[f'{key}_mean'].values[0]) * 100
            seed_vals = (1.0 - seed_vals) * 100
        else:
            mean_val = row[f'{key}_mean'].values[0] * 100
            seed_vals = seed_vals * 100
        means.append(mean_val)
        seeds.append(seed_vals)

    y = np.arange(len(methods))[::-1]
    for yi, m, mu, sv in zip(y, methods, means, seeds):
        c = colors[m]
        # seed-range whisker (min..max) + small per-seed dots
        if len(sv):
            ax.plot([sv.min(), sv.max()], [yi, yi], color=c, linewidth=2.0,
                    alpha=0.55, zorder=2)
            ax.scatter(sv, [yi] * len(sv), s=24, color='white',
                       edgecolor=c, linewidth=1.0, zorder=3)
        ax.scatter(mu, yi, s=110, color=c, edgecolor='black', linewidth=0.8,
                   zorder=4)
        ax.text(mu, yi + 0.32, f'{mu:.1f}', ha='center', va='bottom',
                fontsize=8.5, color='black')

    ax.set_xlabel(title, fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(methods) - 0.3)
    ax.grid(axis='x', linestyle=':', alpha=0.4, zorder=0)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.axvline(0, color='#bbb', linewidth=0.4, zorder=1)

# Shared y tick labels (left axis only)
axes[0].set_yticks(np.arange(len(methods))[::-1])
axes[0].set_yticklabels([method_display[m] for m in methods], fontsize=9.5)

plt.tight_layout()
save_fig(fig, 'fig3_ablation')
plt.close()
