"""Shared plotting style for Probe-Then-Act paper figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

COLORS = {
    'm1_reactive': '#4477AA',
    'm7_pta': '#EE6677',
    'm8_teacher': '#228833',
    'm7_noprobe': '#CCBB44',
    'm7_nobelief': '#AA3377',
}

METHOD_LABELS = {
    'm1_reactive': 'M1: Reactive',
    'm7_pta': 'M7: Probe-Then-Act (Ours)',
    'm8_teacher': 'M8: Privileged Teacher',
    'm7_noprobe': 'M7: No Probe',
    'm7_nobelief': 'M7: No Belief',
}

SPLIT_LABELS = {
    'id_sand': 'ID\nSand',
    'ood_elastoplastic': 'OOD\nElastoplastic',
    'ood_snow': 'OOD\nSnow',
    'ood_sand_hard': 'OOD\nSand (Hard)',
    'ood_sand_soft': 'OOD\nSand (Soft)',
}

SPLIT_ORDER = ['id_sand', 'ood_elastoplastic', 'ood_snow', 'ood_sand_hard', 'ood_sand_soft']

FIG_DIR = 'figures'


def save_fig(fig, name, fmt='pdf'):
    """Save figure to FIG_DIR."""
    path = f'{FIG_DIR}/{name}.{fmt}'
    fig.savefig(path)
    print(f'Saved: {path}')
    return path
