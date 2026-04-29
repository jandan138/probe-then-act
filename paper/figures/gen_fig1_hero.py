"""Figure 1 (Hero): Method overview + cross-material comparison."""
import sys
sys.path.insert(0, 'figures')
from paper_plot_style import *
import pandas as pd
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle

df = pd.read_csv('results/main_results.csv')

fig = plt.figure(figsize=(14, 4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)

# ============================================================
# LEFT PANEL: Method overview (rendered illustration)
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
pipeline_img = plt.imread('figures/fig1_pipeline.png')
ax1.imshow(pipeline_img, aspect='equal')
ax1.axis('off')

# ============================================================
# RIGHT PANEL: Cross-material comparison
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])

methods = ['m1_reactive', 'm7_pta', 'm8_teacher']
splits = SPLIT_ORDER

n_methods = len(methods)
bar_width = 0.26
x = np.arange(len(splits))

for i, method in enumerate(methods):
    means = []
    stds = []
    for split in splits:
        row = df[(df['method'] == method) & (df['split'] == split)]
        if len(row) > 0:
            means.append(row['mean_transfer_mean'].values[0] * 100)
            stds.append(row['mean_transfer_std'].values[0] * 100)
        else:
            means.append(0)
            stds.append(0)
    offset = (i - (n_methods - 1) / 2) * bar_width
    ax2.bar(x + offset, means, bar_width,
            yerr=stds, capsize=2,
            label=METHOD_LABELS[method],
            color=COLORS[method], edgecolor='black', linewidth=0.5,
            error_kw={'linewidth': 0.7, 'ecolor': '#333333'})

# Highlight elastoplastic with box + asterisk
ep_idx = splits.index('ood_elastoplastic')
rect = Rectangle((ep_idx - 0.5, -2), 1.0, 100,
                 linewidth=1.3, edgecolor='#AA3377', facecolor='none',
                 linestyle='--', zorder=0)
ax2.add_patch(rect)
m7_ep_mean = df[(df['method'] == 'm7_pta') & (df['split'] == 'ood_elastoplastic')]['mean_transfer_mean'].values[0] * 100
ax2.annotate('$^*$+14.7pp', xy=(ep_idx, m7_ep_mean + 4),
             ha='center', fontsize=9, fontweight='bold', color='#AA3377')

ax2.set_xticks(x)
ax2.set_xticklabels([SPLIT_LABELS[s] for s in splits], fontsize=8)
ax2.set_ylabel('Transfer Efficiency (\\%)')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left', frameon=False, fontsize=8)
ax2.set_title('(b) Cross-Material Transfer Efficiency', fontsize=11, fontweight='bold')

save_fig(fig, 'fig1_hero')
plt.close()
