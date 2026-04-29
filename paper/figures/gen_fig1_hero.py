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
# LEFT PANEL: Method overview
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5)
ax1.axis('off')

# Stage 1: Probe Phase
probe_box = FancyBboxPatch((0.2, 2.8), 2.4, 1.6,
                           boxstyle="round,pad=0.1",
                           facecolor='#E8F0F8', edgecolor='#4477AA', linewidth=1.5)
ax1.add_patch(probe_box)
ax1.text(1.4, 4.1, 'Probe Phase', ha='center', fontsize=11, fontweight='bold', color='#2C5282')
ax1.text(1.4, 3.6, '3 exploration steps', ha='center', fontsize=9)
ax1.text(1.4, 3.2, r'$\tau_{\mathrm{probe}} = \{(s_t, a_t)\}_{t=0}^{2}$', ha='center', fontsize=9)

# Stage 2: Belief Encoder
belief_box = FancyBboxPatch((3.5, 2.8), 2.6, 1.6,
                            boxstyle="round,pad=0.1",
                            facecolor='#FDF0F0', edgecolor='#EE6677', linewidth=1.5)
ax1.add_patch(belief_box)
ax1.text(4.8, 4.1, 'Belief Encoder', ha='center', fontsize=11, fontweight='bold', color='#B83E54')
ax1.text(4.8, 3.6, r'MLP + Mean-Pool', ha='center', fontsize=9)
ax1.text(4.8, 3.2, r'$\to (z \in \mathbb{R}^{16}, \sigma)$', ha='center', fontsize=9)

# Stage 3: Task Policy
task_box = FancyBboxPatch((7.0, 2.8), 2.6, 1.6,
                          boxstyle="round,pad=0.1",
                          facecolor='#EAFAEE', edgecolor='#228833', linewidth=1.5)
ax1.add_patch(task_box)
ax1.text(8.3, 4.1, 'Task Policy', ha='center', fontsize=11, fontweight='bold', color='#1A5F28')
ax1.text(8.3, 3.6, r'$\pi_\theta(a_t \mid s_t, z, \sigma)$', ha='center', fontsize=9)
ax1.text(8.3, 3.2, r'Joint-residual PPO', ha='center', fontsize=9)

# Arrows between stages
arrow1 = FancyArrowPatch((2.65, 3.6), (3.45, 3.6),
                         arrowstyle='->', mutation_scale=15, linewidth=1.2, color='#444')
ax1.add_patch(arrow1)
arrow2 = FancyArrowPatch((6.15, 3.6), (6.95, 3.6),
                         arrowstyle='->', mutation_scale=15, linewidth=1.2, color='#444')
ax1.add_patch(arrow2)

# Bottom row: materials and probe traces
materials = [
    ('Sand\n(granular)', '#D4A574', 1.0),
    ('Snow\n(cohesive)', '#C8E0E8', 3.1),
    ('Elastoplastic\n(viscoelastic)', '#C4B4D9', 5.2),
]
ax1.text(0.2, 1.6, 'Materials:', fontsize=9, fontweight='bold')
for label, color, x_pos in materials:
    mat_box = FancyBboxPatch((x_pos, 0.5), 1.4, 0.9,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='#333', linewidth=0.8, alpha=0.7)
    ax1.add_patch(mat_box)
    ax1.text(x_pos + 0.7, 0.95, label, ha='center', va='center', fontsize=8)

# Hidden-properties annotation
ax1.annotate('Hidden: $E, \\nu, \\rho$, cohesion',
             xy=(6.9, 0.95), fontsize=9, style='italic', color='#666')

ax1.text(5.0, 5.0, '(a) Probe-Then-Act Pipeline',
         ha='center', fontsize=11, fontweight='bold')

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
