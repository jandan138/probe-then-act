"""Figure 1 (Hero): Pipeline schematic + effect plot (PTA - Reactive deltas).

Left: AI-rendered pipeline schematic (raster).
Right: Forest / dumbbell effect plot of M7-M1 transfer delta in pp,
ranked across the 5 splits, with M8-M1 deltas overlaid as hollow markers.
The plot makes the central PTA thesis (one OOD win, four losses) immediate.
"""
import sys
sys.path.insert(0, 'figures')
from paper_plot_style import *
import pandas as pd
import numpy as np

df = pd.read_csv('results/main_results.csv')

fig = plt.figure(figsize=(14, 4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.18)

# ============================================================
# LEFT PANEL: Method overview (rendered illustration)
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
pipeline_img = plt.imread('figures/fig1_pipeline.png')
ax1.imshow(pipeline_img, aspect='equal')
ax1.axis('off')

# ============================================================
# RIGHT PANEL: Effect plot (M7-M1 deltas, with M8-M1 overlay)
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])

# Order rows so OOD-Elastoplastic (the one positive) is on top, others below.
plot_order = [
    'ood_elastoplastic',
    'ood_snow',
    'ood_sand_hard',
    'ood_sand_soft',
    'id_sand',
]
labels = [SPLIT_LABELS[s].replace('\n', ' ') for s in plot_order]

m1 = df[df['method'] == 'm1_reactive'].set_index('split')['mean_transfer_mean'] * 100
m7 = df[df['method'] == 'm7_pta'].set_index('split')['mean_transfer_mean'] * 100
m8 = df[df['method'] == 'm8_teacher'].set_index('split')['mean_transfer_mean'] * 100

delta_m7 = (m7 - m1).reindex(plot_order)
delta_m8 = (m8 - m1).reindex(plot_order)

y = np.arange(len(plot_order))[::-1]   # top row at top

# Zero line
ax2.axvline(0, color='#444', linewidth=0.8, zorder=1)

# M7-M1 dots (filled): green for positive, muted red for negative
for yi, sp in zip(y, plot_order):
    d = delta_m7[sp]
    color = '#1A8754' if d > 0 else '#C44E52'
    ax2.plot([0, d], [yi, yi], color=color, linewidth=2.2, alpha=0.55, zorder=2)
    ax2.scatter(d, yi, s=80, color=color, edgecolor='black', linewidth=0.7,
                zorder=4, label=None)
    label = f'{d:+.1f}'
    offset = 1.5 if d >= 0 else -1.5
    ha = 'left' if d >= 0 else 'right'
    ax2.text(d + offset, yi, label, ha=ha, va='center', fontsize=8.5,
             color='black', fontweight='bold' if abs(d) > 14 else 'normal')

# M8-M1 hollow triangles
for yi, sp in zip(y, plot_order):
    d = delta_m8[sp]
    ax2.scatter(d, yi - 0.18, s=55, marker='v', facecolor='white',
                edgecolor='#888', linewidth=1.0, zorder=3, label=None)

# Compact, frameless legend; sign is also encoded by marker fill (filled
# circle = M7) and shape (hollow triangle = M8) so colour is redundant.
ax2.scatter([], [], s=80, color='#1A8754', edgecolor='black', linewidth=0.7,
            label='M7 - M1 (gain)')
ax2.scatter([], [], s=80, color='#C44E52', edgecolor='black', linewidth=0.7,
            label='M7 - M1 (loss)')
ax2.scatter([], [], s=55, marker='v', facecolor='white', edgecolor='#888',
            linewidth=1.0, label='M8 - M1')
ax2.legend(loc='lower right', frameon=False, fontsize=8, handletextpad=0.3,
           borderaxespad=0.3, labelspacing=0.25)

ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel('Transfer-efficiency delta vs.\\ M1 Reactive (pp)')
ax2.set_xlim(-50, 25)
ax2.set_ylim(-0.7, len(plot_order) - 0.3)
ax2.spines['left'].set_visible(False)
ax2.tick_params(axis='y', length=0)
ax2.grid(axis='x', linestyle=':', alpha=0.45, zorder=0)
ax2.set_title('Per-split effect (\\PTA{} - Reactive)', fontsize=10,
              fontweight='bold', loc='left', pad=6)

save_fig(fig, 'fig1_hero')
plt.close()
