import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import kendalltau, weightedtau

from graph_metrics import strengths_und, efficiency_wei, charpath


def pvalue_percentages(pvalues, alpha=0.01):
    """

    :param pvalues:
    :param alpha:
    :return:
    """
    fdr = fdr_corrected_percentage(pvalues, alpha)
    fwe = fwe_corrected_percentage(pvalues, alpha)
    nocor = 100 * np.mean(pvalues < alpha)

    return fdr, fwe, nocor


def fdr_corrected_percentage(pvalues, alpha=0.01):
    """

    :param pvalues:
    :param alpha:
    :return:
    """
    m = len(pvalues)
    k_all = np.array(list(range(1, m + 1)))
    reject_list = np.where(
        np.sort(pvalues) <= (k_all * alpha / m)
    )[0]
    if len(reject_list) > 0:
        k = reject_list[-1]
    else:
        k = 0

    return 100 * k / m


def fwe_corrected_percentage(pvalues, alpha=0.01):
    """

    :param pvalues:
    :param alpha:
    :return:
    """
    m = len(pvalues)
    bonf_alpha = alpha / m

    return 100 * np.mean(pvalues < bonf_alpha)


def significant_edges_t(target, source, alpha=0.01):
    """
    Function to check significantly different edges (no check for which is
     higher).
    :param target:
    :param source:
    :param alpha:
    :return:
    """

    fdr_list = []
    fwe_list = []
    nocor_list = []
    pvalues_list = []
    for m in source:
        _, pvalues = ttest_rel(target, m, axis=0)
        novar_mask = np.isnan(pvalues)
        pvalues[novar_mask] = 1

        fdr, fwe, nocor = pvalue_percentages(pvalues, alpha)

        fdr_list.append(fdr)
        fwe_list.append(fwe)
        nocor_list.append(nocor)
        pvalues_list.append(pvalues)

    return pvalues_list, fdr_list, fwe_list, nocor_list


def significant_edges_w(target, source, alpha=0.01):
    """
    Function to check significantly different edges (no check for which is
     higher).
    :param target:
    :param source:
    :param alpha:
    :return:
    """

    fdr_list = []
    fwe_list = []
    nocor_list = []
    pvalues_list = []
    for m in source:
        pvalues = np.array([
            wilcoxon(gs_i, pipe_i, 'zsplit')[1]
            for gs_i, pipe_i in zip(target.transpose(), m.transpose())
        ])
        novar_mask = np.isnan(pvalues)
        pvalues[novar_mask] = 1

        fdr, fwe, nocor = pvalue_percentages(pvalues, alpha)

        fdr_list.append(fdr)
        fwe_list.append(fwe)
        nocor_list.append(nocor)
        pvalues_list.append(pvalues)

    return pvalues_list, fdr_list, fwe_list, nocor_list


def conn_mask(edge_list, edges):
    """

    :param edge_list:
    :param edges:
    :return:
    """
    edge_mask = np.zeros(edges)
    edge_mask[edge_list] = 1

    return triu_to_graph(edge_mask)


def triu_to_graph(triu):
    """

    :param triu:
    :return:
    """
    n_edges = len(triu)
    nodes = int((np.sqrt(8 * n_edges + 1) + 1) / 2)
    r, c = np.triu_indices(nodes, 1)

    graph = np.zeros((nodes, nodes))
    graph[r, c] = triu
    graph[c, r] = triu

    return graph


def graph_tests(target, source):
    relstr_t = []
    releff_t = []
    relcpath_t = []
    relstr_w = []
    releff_w = []
    relcpath_w = []

    gt_strength = np.mean([
        strengths_und(conn) for conn in target
    ], axis=-1)
    gt_efficiency = np.array([
        efficiency_wei(conn) for conn in target
    ])
    gt_charpath = np.array([
        charpath(conn)[0] for conn in target
    ])
    for m in source:
        m_strength = np.mean([
            strengths_und(conn) for conn in m
        ], axis=-1)
        m_efficiency = np.array([
            efficiency_wei(conn) for conn in m
        ], axis=-1)
        m_charpath = np.array([
            charpath(conn)[0] for conn in m
        ], axis=-1)

        _, pvalue = ttest_rel(gt_strength, m_strength)
        relstr_t.append(pvalue)
        _, pvalue = wilcoxon(gt_strength, m_strength)
        relstr_w.append(pvalue)

        _, pvalue = ttest_rel(gt_efficiency, m_efficiency)
        releff_t.append(pvalue)
        _, pvalue = wilcoxon(gt_efficiency, m_efficiency)
        releff_w.append(pvalue)

        _, pvalue = ttest_rel(gt_charpath, m_charpath)
        relcpath_t.append(pvalue)
        _, pvalue = wilcoxon(gt_charpath, m_charpath)
        relcpath_w.append(pvalue)

    return relstr_t, relstr_w, releff_t, releff_w, relcpath_t, relcpath_w


def disparity_matrix(target, source):
    disp = []
    for m in source:
        disp.append(np.mean(np.abs(target - m), axis=0))

    return disp


def ranking_metrics(target, source, alpha=0.01):
    """

    :param target:
    :param source:
    :return:
    """

    m = target.shape[-1]
    bonf_alpha = alpha / m

    # Comparison between the gold standard and other methods.
    inter_taus = []
    inter_wtaus = []
    inter_iwtaus = []
    inter_pvalues = []
    for m in source:
        inter_taus.append(
            [kendalltau(gs_sub, pipe_sub)[0] for gs_sub, pipe_sub in zip(target, m)]
        )
        inter_wtaus.append(
            [weightedtau(gs_sub, pipe_sub)[0] for gs_sub, pipe_sub in zip(target, m)]
        )
        inter_iwtaus.append(
            [weightedtau(-gs_sub, -pipe_sub)[0] for gs_sub, pipe_sub in zip(target, m)]
        )
        inter_pvalues.append(
            np.mean([
                kendalltau(gs_sub, pipe_sub)[1] < bonf_alpha
                for gs_sub, pipe_sub in zip(target, m)
            ])
        )

    inter_corr = [inter_taus, inter_wtaus, inter_iwtaus, inter_pvalues]

    # Comparison between gold standard inviduals.
    # This is a bit different from the previous set of metrics. Here we want
    # to see how correlated are the rankings between subjects. Ideally, this
    # should be lower than the correlation between methods. Otherwise, the
    # "errors" would be higher than the real variability between individuals.
    intra_taus = []
    intra_wtaus = []
    intra_iwtaus = []
    intra_pvalues = []
    for i in range(0, len(target)):
        for j in range(i + 1, len(target)):
            i_conn = target[i, :]
            j_conn = target[j, :]
            tau, pvalue = kendalltau(i_conn, j_conn)
            intra_taus.append(tau)
            intra_pvalues.append(pvalue < bonf_alpha)
            tau, _ = weightedtau(i_conn, j_conn)
            intra_wtaus.append(tau)
            tau, _ = weightedtau(-i_conn, -j_conn)
            intra_iwtaus.append(tau)

    intra_corr = [intra_taus, intra_wtaus, intra_iwtaus, intra_pvalues]

    return inter_corr, intra_corr
