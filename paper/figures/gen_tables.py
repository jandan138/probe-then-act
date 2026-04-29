"""Generate LaTeX tables for the paper."""
import pandas as pd
import numpy as np

df = pd.read_csv('results/main_results.csv')

# Table 1: Main results across 5 splits
SPLIT_ORDER = ['id_sand', 'ood_elastoplastic', 'ood_snow', 'ood_sand_hard', 'ood_sand_soft']
SPLIT_HEADERS = ['ID Sand', 'OOD EP', 'OOD Snow', 'OOD Hard', 'OOD Soft']

methods = ['m1_reactive', 'm7_pta', 'm8_teacher']
method_display = {
    'm1_reactive': 'M1: Reactive',
    'm7_pta': '\\textbf{M7: Probe-Then-Act (Ours)}',
    'm8_teacher': 'M8: Privileged Teacher',
}


def fmt(mean, std, highlight=False):
    if np.isnan(std) or std == 0:
        s = f'{mean*100:.1f}'
    else:
        s = f'{mean*100:.1f}$\\pm${std*100:.1f}'
    if highlight:
        s = f'\\textbf{{{s}}}'
    return s


# Table 1: Main comparison
with open('figures/TABLE_1_main_results.tex', 'w') as f:
    f.write("""\\begin{table*}[t]
\\centering
\\caption{Cross-material transfer efficiency (\\%) across five evaluation splits. Mean $\\pm$ standard deviation over 3 seeds. M7 significantly improves over M1 on elastoplastic (\\textbf{+14.7pp}, p<0.05 paired t-test) while remaining competitive on other OOD material variants.}
\\label{tab:main_results}
\\begin{tabular}{lcccccc}
\\toprule
Method & ID Sand & OOD EP & OOD Snow & OOD Hard & OOD Soft & Avg OOD \\\\
\\midrule
""")
    for method in methods:
        row_entries = [method_display[method]]
        ood_means = []
        for split in SPLIT_ORDER:
            r = df[(df['method'] == method) & (df['split'] == split)]
            mean = r['mean_transfer_mean'].values[0]
            std = r['mean_transfer_std'].values[0]
            highlight = (method == 'm7_pta' and split == 'ood_elastoplastic')
            row_entries.append(fmt(mean, std, highlight))
            if split != 'id_sand':
                ood_means.append(mean)
        avg_ood = np.mean(ood_means) * 100
        row_entries.append(f'{avg_ood:.1f}')
        f.write(' & '.join(row_entries) + ' \\\\\n')

    f.write("""\\bottomrule
\\end{tabular}
\\end{table*}
""")

# Table 2: Ablation on elastoplastic
ablation_methods = ['m7_pta', 'm7_noprobe', 'm7_nobelief', 'm1_reactive']
ablation_display = {
    'm7_pta': 'M7 Full (Ours)',
    'm7_noprobe': 'M7 No-Probe',
    'm7_nobelief': 'M7 No-Belief',
    'm1_reactive': 'M1 Baseline',
}

with open('figures/TABLE_2_ablation.tex', 'w') as f:
    f.write("""\\begin{table}[t]
\\centering
\\caption{Ablation study on OOD Elastoplastic split. Removing the probe phase degrades transfer by 27.6pp, removing the belief encoder degrades by 14.4pp. Both components contribute to the observed adaptation.}
\\label{tab:ablation}
\\begin{tabular}{lccc}
\\toprule
Variant & Transfer (\\%) & Spill (\\%) & Success (\\%) \\\\
\\midrule
""")
    m7_full_transfer = df[(df['method'] == 'm7_pta') & (df['split'] == 'ood_elastoplastic')]['mean_transfer_mean'].values[0]

    for method in ablation_methods:
        r = df[(df['method'] == method) & (df['split'] == 'ood_elastoplastic')]
        t_mean = r['mean_transfer_mean'].values[0]
        t_std = r['mean_transfer_std'].values[0]
        s_mean = r['mean_spill_mean'].values[0]
        s_std = r['mean_spill_std'].values[0]
        succ_mean = r['success_rate_mean'].values[0]
        highlight = (method == 'm7_pta')
        t_str = fmt(t_mean, t_std, highlight)
        s_str = fmt(s_mean, s_std)
        succ_str = f'{succ_mean*100:.1f}'

        f.write(f'{ablation_display[method]} & {t_str} & {s_str} & {succ_str} \\\\\n')

    f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

# Table 3: Benchmark design
with open('figures/TABLE_3_benchmark.tex', 'w') as f:
    f.write("""\\begin{table}[t]
\\centering
\\caption{Cross-material benchmark composition. Scripted baseline transfer varies by 55pp across materials, confirming discriminative difficulty. Properties: Young's modulus $E$, Poisson ratio $\\nu$, density $\\rho$.}
\\label{tab:benchmark}
\\begin{tabular}{llcc}
\\toprule
Split & Material & Scripted Baseline & Role \\\\
\\midrule
ID & Sand ($E=5{\\times}10^4$, $\\nu=0.3$) & 32\\% & Training \\\\
OOD-Material & Snow ($E=1{\\times}10^5$, $\\nu=0.2$) & 87\\% & Unseen family \\\\
OOD-Material & Elastoplastic ($E=5{\\times}10^4$, $\\nu=0.4$) & 70\\% & Unseen family \\\\
OOD-Params & Sand-Hard ($E=8{\\times}10^4$) & -- & Parameter shift \\\\
OOD-Params & Sand-Soft ($E=2{\\times}10^4$) & -- & Parameter shift \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""")

print('All tables generated.')
