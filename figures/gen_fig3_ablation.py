"""Figure 3: Ablation study on elastoplastic split."""
import sys
sys.path.insert(0, 'figures')
from paper_plot_style import *
import pandas as pd
import numpy as np

df = pd.read_csv('results/main_results.csv')

methods = ['m7_pta', 'm7_noprobe', 'm7_nobelief', 'm1_reactive']
method_display = {
    'm7_pta': 'Full (Ours)',
    'm7_noprobe': 'No Probe',
    'm7_nobelief': 'No Belief',
    'm1_reactive': 'M1 Baseline',
}

ep_rows = df[df['split'] == 'ood_elastoplastic']

transfer_means = []
transfer_stds = []
spill_means = []
spill_stds = []
success_means = []

for method in methods:
    row = ep_rows[ep_rows['method'] == method]
    transfer_means.append(row['mean_transfer_mean'].values[0] * 100)
    transfer_stds.append(row['mean_transfer_std'].values[0] * 100)
    spill_means.append(row['mean_spill_mean'].values[0] * 100)
    spill_stds.append(row['mean_spill_std'].values[0] * 100)
    success_means.append(row['success_rate_mean'].values[0] * 100)

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=False)

x = np.arange(len(methods))
bar_colors = [COLORS['m7_pta'], COLORS['m7_noprobe'], COLORS['m7_nobelief'], COLORS['m1_reactive']]

# Transfer
ax = axes[0]
bars = ax.bar(x, transfer_means, yerr=transfer_stds, capsize=3,
              color=bar_colors, edgecolor='black', linewidth=0.5,
              error_kw={'linewidth': 0.8, 'ecolor': '#333333'})
for bar, val in zip(bars, transfer_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([method_display[m] for m in methods], rotation=20, ha='right')
ax.set_ylabel('Transfer Efficiency (\\%)')
ax.set_ylim(0, 100)
ax.set_title('(a) Transfer Efficiency', fontsize=10)

# Spill
ax = axes[1]
bars = ax.bar(x, spill_means, yerr=spill_stds, capsize=3,
              color=bar_colors, edgecolor='black', linewidth=0.5,
              error_kw={'linewidth': 0.8, 'ecolor': '#333333'})
for bar, val in zip(bars, spill_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([method_display[m] for m in methods], rotation=20, ha='right')
ax.set_ylabel('Spill Ratio (\\%)')
ax.set_ylim(0, 100)
ax.set_title('(b) Spill Ratio (lower is better)', fontsize=10)

# Success
ax = axes[2]
bars = ax.bar(x, success_means,
              color=bar_colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, success_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([method_display[m] for m in methods], rotation=20, ha='right')
ax.set_ylabel('Success Rate (\\%)')
ax.set_ylim(0, 100)
ax.set_title('(c) Success Rate', fontsize=10)

plt.tight_layout()
save_fig(fig, 'fig3_ablation')
plt.close()
