"""Generate the workflow and per-split delta figures.

The NeurIPS paper uses the workflow and delta plot as separate figures.  The
legacy combined hero PDF is preserved for the IEEE venue and is only regenerated
when GENERATE_LEGACY_HERO is enabled.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
from paper_plot_style import *
import pandas as pd
import numpy as np

df = pd.read_csv(ROOT / 'results' / 'main_results.csv')
GENERATE_LEGACY_HERO = False

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


def save_shared(fig, name):
    path = SCRIPT_DIR / f'{name}.pdf'
    fig.savefig(path)
    print(f'Saved: {path}')
    return path


WORKFLOW_TITLE_CROP_PX = 72


def load_workflow_image(crop_title=False):
    pipeline_img = plt.imread(SCRIPT_DIR / 'fig1_pipeline.png')
    if crop_title:
        pipeline_img = pipeline_img[WORKFLOW_TITLE_CROP_PX:, :, :]
    return pipeline_img


def add_workflow(ax, crop_title=False):
    pipeline_img = load_workflow_image(crop_title=crop_title)
    ax.imshow(pipeline_img, aspect='equal')
    ax.axis('off')


def add_delta_plot(ax):
    y = np.arange(len(plot_order))[::-1]

    ax.axvline(0, color='#444', linewidth=0.8, zorder=1)

    for yi, sp in zip(y, plot_order):
        d = delta_m7[sp]
        color = '#1A8754' if d > 0 else '#C44E52'
        ax.plot([0, d], [yi, yi], color=color, linewidth=2.2,
                alpha=0.55, zorder=2)
        ax.scatter(d, yi, s=80, color=color, edgecolor='black',
                   linewidth=0.7, zorder=4, label=None)
        offset = 1.5 if d >= 0 else -1.5
        ha = 'left' if d >= 0 else 'right'
        ax.text(d + offset, yi, f'{d:+.1f}', ha=ha, va='center',
                fontsize=8.5, color='black',
                fontweight='bold' if abs(d) > 14 else 'normal')

    for yi, sp in zip(y, plot_order):
        d = delta_m8[sp]
        ax.scatter(d, yi - 0.18, s=55, marker='v', facecolor='white',
                   edgecolor='#888', linewidth=1.0, zorder=3, label=None)

    ax.scatter([], [], s=80, color='#1A8754', edgecolor='black',
               linewidth=0.7, label='M7 - M1 gain')
    ax.scatter([], [], s=80, color='#C44E52', edgecolor='black',
               linewidth=0.7, label='M7 - M1 loss')
    ax.scatter([], [], s=55, marker='v', facecolor='white', edgecolor='#888',
               linewidth=1.0, label='M8 - M1')
    ax.legend(loc='lower right', frameon=False, fontsize=8, handletextpad=0.3,
              borderaxespad=0.3, labelspacing=0.25)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Transfer-efficiency delta vs. M1 Reactive (pp)')
    ax.set_xlim(-50, 25)
    ax.set_ylim(-0.7, len(plot_order) - 0.3)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', linestyle=':', alpha=0.45, zorder=0)


# Standalone workflow figure for the main text.
pipeline_img = load_workflow_image(crop_title=True)
height, width = pipeline_img.shape[:2]
workflow_width = 10.0
workflow_height = workflow_width * height / width
fig_workflow, ax_workflow = plt.subplots(figsize=(workflow_width, workflow_height))
add_workflow(ax_workflow, crop_title=True)
fig_workflow.subplots_adjust(left=0, right=1, top=1, bottom=0)
save_shared(fig_workflow, 'fig1_workflow')
plt.close(fig_workflow)


# Standalone delta plot for the Results section.
fig_delta, ax_delta = plt.subplots(figsize=(6.4, 3.0))
add_delta_plot(ax_delta)
save_shared(fig_delta, 'fig1_effect_delta')
plt.close(fig_delta)


if GENERATE_LEGACY_HERO:
    # Legacy combined hero figure for venues that still use it.
    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    add_workflow(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    add_delta_plot(ax2)

    save_shared(fig, 'fig1_hero')
    plt.close()
