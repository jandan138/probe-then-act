"""Figure 2: Main cross-material comparison (M1 vs M7 vs M8)."""
import sys
sys.path.insert(0, 'figures')
from paper_plot_style import *
import pandas as pd
import numpy as np

df = pd.read_csv('results/main_results.csv')

methods = ['m1_reactive', 'm7_pta', 'm8_teacher']
splits = SPLIT_ORDER

fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

n_methods = len(methods)
bar_width = 0.25
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
    bars = ax.bar(x + offset, means, bar_width,
                  yerr=stds, capsize=3,
                  label=METHOD_LABELS[method],
                  color=COLORS[method], edgecolor='black', linewidth=0.5,
                  error_kw={'linewidth': 0.8, 'ecolor': '#333333'})

# Highlight elastoplastic with asterisk
ep_idx = splits.index('ood_elastoplastic')
m7_ep_mean = df[(df['method'] == 'm7_pta') & (df['split'] == 'ood_elastoplastic')]['mean_transfer_mean'].values[0] * 100
ax.annotate('*', xy=(ep_idx, m7_ep_mean + 3), ha='center', fontsize=14, fontweight='bold', color='#EE6677')

ax.set_xticks(x)
ax.set_xticklabels([SPLIT_LABELS[s] for s in splits])
ax.set_ylabel('Transfer Efficiency (\\%)')
ax.set_ylim(0, 95)
ax.legend(loc='upper right', frameon=False, ncol=1)
ax.axhline(y=30, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.text(4.3, 32, 'Success threshold', fontsize=7, color='gray', style='italic')

save_fig(fig, 'fig2_main_comparison')
plt.close()
