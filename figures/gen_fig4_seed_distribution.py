"""Figure 4: Per-seed distribution showing material-specific gain."""
import sys
sys.path.insert(0, 'figures')
from paper_plot_style import *
import pandas as pd
import numpy as np

df = pd.read_csv('results/ood_eval_per_seed.csv')

methods = ['m1_reactive', 'm7_pta']
splits = SPLIT_ORDER

fig, ax = plt.subplots(1, 1, figsize=(7, 3.2))

x = np.arange(len(splits))
bar_width = 0.35

for i, method in enumerate(methods):
    seed_values = []
    for split in splits:
        vals = df[(df['method'] == method) & (df['split'] == split)]['mean_transfer'].values * 100
        seed_values.append(vals)

    offset = (i - 0.5) * bar_width
    means = [v.mean() for v in seed_values]
    bars = ax.bar(x + offset, means, bar_width,
                  label=METHOD_LABELS[method],
                  color=COLORS[method], edgecolor='black', linewidth=0.5, alpha=0.85)

    # Overlay individual seeds as dots
    for j, vals in enumerate(seed_values):
        ax.scatter([x[j] + offset] * len(vals), vals,
                   s=18, color='white', edgecolor='black', linewidth=0.6, zorder=3)

# Highlight elastoplastic gain
ep_idx = splits.index('ood_elastoplastic')
ax.annotate('', xy=(ep_idx + 0.18, 67), xytext=(ep_idx - 0.18, 47),
            arrowprops=dict(arrowstyle='->', color='#AA3377', lw=1.3))
ax.text(ep_idx + 0.4, 58, '+14.7 pp', fontsize=9,
        color='#AA3377', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([SPLIT_LABELS[s] for s in splits])
ax.set_ylabel('Transfer Efficiency (\\%)')
ax.set_ylim(0, 100)
ax.legend(loc='upper right', frameon=False)

save_fig(fig, 'fig4_seed_distribution')
plt.close()
