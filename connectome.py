import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
import csv
import numpy as np
import pandas as pd
import seaborn as sn
import nibabel as nib
import statsmodels.api as smapi
from scipy.stats import ttest_ind, mannwhitneyu, normaltest, ttest_rel, wilcoxon
from scipy.stats import kendalltau, weightedtau


m = 84
r, c = np.triu_indices(m, 1)

connectomes = []
for sub in subjects:
    sub_conn = []
    sub_available = True
    for pipe_type in types:
        try:
            conn = np.genfromtxt(
                find_file(sub[:3], os.path.join(path, pipe_type)),
                delimiter=','
            )
            sub_conn.append(conn)
        except TypeError:
            sub_available = False
    if sub_available:
        connectomes.append(sub_conn)

connectomes = np.array(connectomes)
triu = connectomes[..., r, c]
print(types, connectomes.shape, triu.shape)


plt.figure(figsize=(15, 12))
plot_correlation(triu[:, 1, :].flatten(), triu[:, 0, :].flatten(), types[1], types[0])

pvalues = {
    pipe: []
    for pipe in types
}
pvalues_w = {
    pipe: []
    for pipe in types
}

nodes = 84
alpha = 0.01
n_edges = (nodes - 1) * nodes / 2
bonf_alpha = alpha / n_edges

fdr_list = []
fwe_list = []
nocor_list = []
fdr_list_w = []
fwe_list_w = []
nocor_list_w = []

for pipe_i in range(0, len(types)):
    pipe = types[pipe_i]
    gs_conn = triu[:, 0, :]
    pipe_conn = triu[:, pipe_i, :]
    stat, pvalue = ttest_rel(gs_conn, pipe_conn, axis=0)
    novar_mask = np.isnan(pvalue)
    pvalue[novar_mask] = 1
    pvalues[pipe] = pvalue
    pvalue = np.array([
        wilcoxon(gs_i, pipe_i, 'zsplit')[1]
        for gs_i, pipe_i in zip(gs_conn.transpose(), pipe_conn.transpose())
    ])
    novar_mask = np.isnan(pvalue)
    pvalue[novar_mask] = 1
    pvalues_w[pipe] = pvalue

for pipe in types[0:]:
    pipe_pvalues = np.sort(pvalues[pipe])
    m = len(pipe_pvalues)
    k_all = np.array(list(range(1, m + 1)))
    reject_list = np.where(pipe_pvalues <= (k_all * alpha / m))[0]
    if len(reject_list) > 0:
        k = reject_list[-1]
    else:
        k = 0
    fdr_list.append(100 * k / m)
    fwe_list.append(100 * np.mean(pvalues[pipe] < bonf_alpha))
    nocor_list.append(100 * np.mean(pvalues[pipe] < alpha))

    pipe_pvalues = np.sort(pvalues_w[pipe])
    m = len(pipe_pvalues)
    k_all = np.array(list(range(1, m + 1)))
    reject_list = np.where(pipe_pvalues <= (k_all * alpha / m))[0]
    if len(reject_list) > 0:
        k = reject_list[-1]
    else:
        k = 0
    fdr_list_w.append(100 * k / m)
    fwe_list_w.append(100 * np.mean(pvalues_w[pipe] < bonf_alpha))
    nocor_list_w.append(100 * np.mean(pvalues_w[pipe] < alpha))

fdr_string = ' | '.join([
    '{:06.2f}%      '.format(p)
    for p in fdr_list
])

fwe_string = ' | '.join([
    '{:06.2f}%      '.format(p)
    for p in fwe_list
])

nocor_string = ' | '.join([
    '{:06.2f}%      '.format(p)
    for p in nocor_list
])

print(
    'Metric                  |', ' | '.join(
        ['{:13}'.format(t) for t in types]
    )
)
print('------------------------|-' + '-|-'.join(['-' * 13] * (len(types))))
print('Edge diff [FDR t]       |', fdr_string)
print('------------------------|-' + '-|-'.join(['-' * 13] * (len(types))))
print('Edge diff [FWE t]       |', fwe_string)
print('------------------------|-' + '-|-'.join(['-' * 13] * (len(types))))
print('Edge diff [t-test]      |', nocor_string)

fdr_w_string = ' | '.join([
    '{:06.2f}%      '.format(p)
    for p in fdr_list_w
])

fwe_w_string = ' | '.join([
    '{:06.2f}%      '.format(p)
    for p in fwe_list_w
])

nocor_w_string = ' | '.join([
    '{:06.2f}%      '.format(p)
    for p in nocor_list_w
])

print('------------------------|-' + '-|-'.join(['-' * 13] * (len(types))))
print('------------------------|-' + '-|-'.join(['-' * 13] * (len(types))))
print('Edge diff [FDR W]       |', fdr_w_string)
print('------------------------|-' + '-|-'.join(['-' * 13] * (len(types))))
print('Edge diff [FWE W]       |', fwe_w_string)
print('------------------------|-' + '-|-'.join(['-' * 13] * (len(types))))
print('Edge diff [Wilcoxon]    |', nocor_w_string)

# pre_pipe = 'Test'
# m12_pipe = 'Retest'
pre_pipe = 'Retest'
m12_pipe = 'Ablation'
alpha = 0.01


def conn_mask(edge_list, edges):
    nodes = int((np.sqrt(8 * edges + 1) + 1) / 2)
    r, c = np.triu_indices(nodes, 1)
    edge_mask = np.zeros(edges)
    edge_mask[edge_list] = 1

    conn_mask = np.zeros((nodes, nodes))
    conn_mask[r, c] = edge_mask
    conn_mask[c, r] = edge_mask

    return conn_mask


def significant_differences(pvalues, alpha):
    n_comparisons = len(pvalues)
    bonf_alpha = alpha / n_comparisons
    sorted_pvalues = np.sort(pvalues)
    sorted_edges = np.argsort(pvalues)
    k_all = np.array(list(range(1, n_comparisons + 1)))
    reject_list = np.where(sorted_pvalues <= (k_all * alpha / m))[0]
    if len(reject_list) > 0:
        fdr_k = reject_list[-1]
    else:
        fdr_k = 0
    fwe_mask = pvalues <= bonf_alpha
    fwe_k = np.sum(fwe_mask)
    fdr_edges = sorted_edges[:fdr_k]
    fwe_edges = np.where(fwe_mask)[0]

    return fdr_k, fwe_k, fdr_edges, fwe_edges


edges = len(pvalues[pre_pipe])

fdr_pre, fwe_pre, diff_pre, _ = significant_differences(pvalues[pre_pipe], alpha)
fdr_12m, fwe_12m, diff_12m, _ = significant_differences(pvalues[m12_pipe], alpha)

plot_conn_masks(1 - pvalues[m12_pipe], np.ones_like(pvalues[m12_pipe]).astype(bool), m12_pipe, 0.99)
plt.suptitle('t-test significance')

thalamus_list = [36, 43]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sn.heatmap(
    np.mean(connectomes, axis=0)[0], xticklabels=False, yticklabels=False,
    square=True, cmap='jet', vmin=-200, vmax=200
)
print(np.mean(connectomes, axis=0)[0].min(), np.mean(connectomes, axis=0)[0].max())
plt.ylabel(pre_pipe)
plt.subplot(1, 2, 2)
sn.heatmap(
    np.mean(connectomes, axis=0)[1], xticklabels=False, yticklabels=False,
    square=True, cmap='jet', vmin=-200, vmax=200
)
print(np.mean(connectomes, axis=0)[1].min(), np.mean(connectomes, axis=0)[1].max())
plt.ylabel(m12_pipe)
plt.suptitle('Average delta connectomes')

plt.figure(figsize=(12, 1))
plt.subplot(1, 2, 1)
connection_list = []
for node in thalamus_list:
    connection_list.append(np.mean(connectomes, axis=0)[0][node, :])
sn.heatmap(
    connection_list, xticklabels=False, yticklabels=False,
    cmap='jet', vmin=-200, vmax=200
)
plt.ylabel(pre_pipe)
plt.subplot(1, 2, 2)
connection_list = []
for node in thalamus_list:
    connection_list.append(np.mean(connectomes, axis=0)[1][node, :])
sn.heatmap(
    connection_list, xticklabels=False, yticklabels=False,
    cmap='jet', vmin=-200, vmax=200
)
plt.ylabel(m12_pipe)
plt.suptitle('Average delta thalamus connections')

print(triu[:, 0, :].shape)

relstr_nonpvalues = []
releff_nonpvalues = []
relstr_pvalues = []
releff_pvalues = []
releff_other_pvalues = []

for pipe in range(1, len(types)):
    gs_conn = strength[:, 0]
    pipe_conn = strength[:, pipe]
    stat, pvalue = ttest_rel(gs_conn, pipe_conn)
    relstr_pvalues.append(pvalue)
    stat, pvalue = wilcoxon(gs_conn, pipe_conn)
    relstr_nonpvalues.append(pvalue)

    gs_conn = efficiency[:, 0]
    pipe_conn = efficiency[:, pipe]
    stat, pvalue = ttest_rel(gs_conn, pipe_conn)
    releff_pvalues.append(pvalue)
    stat, pvalue = wilcoxon(gs_conn, pipe_conn)
    releff_nonpvalues.append(pvalue)

str_pval = ' | '.join([
    '{:4.2e} ({:4.2e})'.format(p, np)
    for p, np in zip(relstr_pvalues, relstr_nonpvalues)
])
eff_pval = ' | '.join([
    '{:4.2e} ({:4.2e})'.format(p, np)
    for p, np in zip(releff_pvalues, releff_nonpvalues)
])

print(
    'Metric             |', ' | '.join(
        ['{:19}'.format(t) for t in types[1:]]
    )
)
print('-------------------|-' + '-|-'.join(['-' * 19] * (len(types) - 1)))
print('Strength           |', str_pval)
print('-------------------|-' + '-|-'.join(['-' * 19] * (len(types) - 1)))
print('Efficiency         |', eff_pval)

m = 84
r, c = np.triu_indices(m, 1)

disp = []
for pipe in range(1, len(types)):
    gs_conn = connectomes[:, 0, ...]
    pipe_conn = connectomes[:, pipe, ...]
    disp.append(np.mean(np.abs(gs_conn - pipe_conn), axis=0))

plt.rcParams.update({'font.size': 12, 'lines.linewidth': 3})
fig = plt.figure(figsize=(16, (len(types) - 1) * 4))
for pipe, d_conn in enumerate(disp):
    max_d = np.max(disp)
    plt.subplot(1, len(types) - 1, pipe + 1)
    ax = sn.heatmap(d_conn, cmap='jet', cbar=False, vmin=0, vmax=max_d, square=True)
    ax.tick_params(right=False, top=False, left=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('{:} ({:e})'.format(types[pipe + 1], np.mean(d_conn[r, c])))

dconn_string = ' | '.join([
    '{:5.3e}    '.format(np.sum(d_conn[r, c])) for d_conn in disp
])
mdconn_string = ' | '.join([
    '{:5.3e}    '.format(np.mean(d_conn[r, c])) for d_conn in disp
])

print(
    'Metric             |', ' | '.join(
        ['{:13}'.format(t) for t in types[1:]]
    )
)
print('-------------------|-' + '-|-'.join(['-' * 13] * (len(types) - 1)))
print('Disparity          |', dconn_string)
print('-------------------|-' + '-|-'.join(['-' * 13] * (len(types) - 1)))
print('µDisparity         |', mdconn_string)

inter_taus = []
inter_wtaus = []
inter_iwtaus = []
inter_pvalues = []
for pipe in range(1, len(types)):
    gs_conn = triu[:, 0, :]
    pipe_conn = triu[:, pipe, :]
    inter_taus.append(
        [kendalltau(gs_sub, pipe_sub)[0] for gs_sub, pipe_sub in zip(gs_conn, pipe_conn)]
    )
    inter_wtaus.append(
        [weightedtau(gs_sub, pipe_sub)[0] for gs_sub, pipe_sub in zip(gs_conn, pipe_conn)]
    )
    inter_iwtaus.append(
        [weightedtau(-gs_sub, -pipe_sub)[0] for gs_sub, pipe_sub in zip(gs_conn, pipe_conn)]
    )
    inter_pvalues.append(
        np.mean([
            kendalltau(gs_sub, pipe_sub)[1] < bonf_alpha
            for gs_sub, pipe_sub in zip(gs_conn, pipe_conn)
        ])
    )

intra_taus = []
intra_wtaus = []
intra_iwtaus = []
intra_pvalues = []
pipe = 0
for i in range(0, len(triu)):
    for j in range(i + 1, len(triu)):
        i_conn = triu[i, 0, :]
        j_conn = triu[j, 0, :]
        tau, pvalue = kendalltau(i_conn, j_conn)
        intra_taus.append(tau)
        intra_pvalues.append(pvalue < bonf_alpha)
        tau, _ = weightedtau(i_conn, j_conn)
        intra_wtaus.append(tau)
        tau, _ = weightedtau(-i_conn, -j_conn)
        intra_iwtaus.append(tau)

tau_string = ' | '.join([
                            '{:4.2f} ± {:4.2f} ({:3.1f})'.format(np.mean(tau), np.std(tau), pvalue)
                            for tau, pvalue in zip(inter_taus, inter_pvalues)
                        ] + [
                            '{:4.2f} ± {:4.2f} ({:3.1f})'.format(
                                np.mean(intra_taus), np.std(intra_taus), np.mean(intra_pvalues)
                            )
                        ])
wtau_string = ' | '.join([
                             '{:4.2f} ± {:4.2f} (-)  '.format(np.mean(tau), np.std(tau))
                             for tau in inter_wtaus
                         ] + [
                             '{:4.2f} ± {:4.2f} (-)'.format(
                                 np.mean(intra_wtaus), np.std(intra_wtaus)
                             )
                         ])
iwtau_string = ' | '.join([
                              '{:4.2f} ± {:4.2f} (-)  '.format(np.mean(tau), np.std(tau))
                              for tau in inter_iwtaus
                          ] + [
                              '{:4.2f} ± {:4.2f} (-)'.format(
                                  np.mean(intra_iwtaus), np.std(intra_iwtaus)
                              )
                          ])
print(
    'Metric      |', ' | '.join(
        ['{:17}'.format(t) for t in types[1:]] + ['Intersubject']
    )
)
print('------------|-' + '-|-'.join(['-' * 17] * (len(types))))
print('τ           |', tau_string)
print('------------|-' + '-|-'.join(['-' * 17] * (len(types))))
print('W           |', wtau_string)
print('------------|-' + '-|-'.join(['-' * 17] * (len(types))))
print('-W          |', iwtau_string)