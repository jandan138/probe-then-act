"""Figure 2: 3 x 5 transfer-efficiency heatmap, full absolute landscape.

Replaces the old grouped bar chart (which duplicated Fig 1 right).
Rows = methods (M1 Reactive, M7 PTA, M8 Privileged-Param).
Cols = 5 material splits (ID Sand, OOD Sand-Hard, OOD Sand-Soft, OOD Snow,
       OOD Elastoplastic).
Cell colour = transfer efficiency %; cell text = numeric value.
The OOD-Elastoplastic column is highlighted to signal the central finding.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
from paper_plot_style import *
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

df = pd.read_csv(ROOT / 'results' / 'main_results.csv')

methods = ['m1_reactive', 'm7_pta', 'm8_teacher']
# Plain text only -- matplotlib has text.usetex=False so LaTeX macros
# would render literally. M8's single-seed status is explained in the caption.
method_labels = ['M1 Reactive', 'M7 PTA (Ours)', 'M8 Privileged-Param']

splits = list(SPLIT_ORDER)   # canonical order: id_sand, EP, snow, hard, soft
split_labels = ['ID\nSand', 'OOD\nElastoplastic', 'OOD\nSnow', 'OOD\nSand-Hard', 'OOD\nSand-Soft']

mat = np.zeros((len(methods), len(splits)))
for i, m in enumerate(methods):
    for j, s in enumerate(splits):
        v = df[(df['method'] == m) & (df['split'] == s)]['mean_transfer_mean'].values
        mat[i, j] = v[0] * 100 if len(v) else np.nan

fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.5), constrained_layout=True)
im = ax.imshow(mat, cmap='YlGnBu', vmin=0, vmax=80, aspect='auto')

# Cell text
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        text_color = 'white' if v > 40 else 'black'
        ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                fontsize=10, color=text_color, fontweight='bold')

# Highlight OOD elastoplastic column
ep_col = splits.index('ood_elastoplastic')
ax.add_patch(Rectangle((ep_col - 0.5, -0.5), 1.0, len(methods),
                       fill=False, edgecolor='#AA3377', linewidth=2.0, zorder=5))

ax.set_xticks(np.arange(len(splits)))
ax.set_xticklabels(split_labels, fontsize=9)
ax.set_yticks(np.arange(len(methods)))
ax.set_yticklabels(method_labels, fontsize=9.5)
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('Transfer efficiency (%)', fontsize=9, rotation=270, labelpad=13)
cbar.ax.tick_params(labelsize=8)

fig.savefig(SCRIPT_DIR / 'fig2_main_comparison.pdf')
print(f"Saved: {SCRIPT_DIR / 'fig2_main_comparison.pdf'}")
plt.close()
