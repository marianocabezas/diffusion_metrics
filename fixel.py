import numpy as np
import time
from copy import deepcopy
from utils import print_progress


def index_to_gt(gt, others):
    n_gt = len(gt)
    indices = []
    errors = []
    for m_i, method in enumerate(others):
        n_method = len(method)
        v1_u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
        v2_u = method / np.linalg.norm(method, axis=1, keepdims=True)
        angle_distance = np.arccos(
            np.clip(np.matmul(v1_u, v2_u.transpose()), -1.0, 1.0)
        )
        angle_index = -np.ones(n_gt, dtype=int)
        angular_error = -np.ones(n_gt, dtype=np.float32)
        indices.append(angle_index)
        errors.append(angular_error)
        if n_method > 0:
            mask = np.zeros((n_gt, n_method))
            masked_distance = np.ma.array(angle_distance, mask=mask)
            for i in range(n_gt):
                if masked_distance.count():
                    flat_index = np.ma.argmin(masked_distance)
                    row, col = np.unravel_index(flat_index, angle_distance.shape)
                    angle_index[row] = col
                    angular_error[row] = angle_distance[row, col]
                    mask[row, :] = 1
                    mask[:, col] = 1
                    masked_distance = np.ma.array(masked_distance, mask=mask)
    return indices, errors


def fixel_comparison(
        gt_indices, gt_peak, gt_afd, gt_dir,
        m_indices, m_peak, m_afd, m_dir
):
    # afd_errors = [[] for m_index in m_indices]
    angular_errors = [0. for _ in m_indices]
    afd_errors = [
        [
            np.zeros(min(gt_i, m_i)) for gt_i, m_i in zip(gt_indices, m_index)
        ]
        for m_index in m_indices
    ]
    missing_afd_errors = [
        [
            np.zeros(gt_i - m_i) if gt_i > m_i else np.zeros(1)
            for gt_i, m_i in zip(gt_indices, m_index)
        ]
        for m_index in m_indices
    ]
    extra_afd_errors = [
        [
            np.zeros(m_i - gt_i) if gt_i < m_i else np.zeros(1)
            for gt_i, m_i in zip(gt_indices, m_index)
        ]
        for m_index in m_indices
    ]
    peak_errors = deepcopy(afd_errors)
    missing_peak_errors = deepcopy(missing_afd_errors)
    extra_peak_errors = deepcopy(extra_afd_errors)
    counts = np.zeros(len(m_indices), dtype=np.float64)
    all_indices = tuple([gt_indices] + m_indices)
    t_init = time.time()
    for index_i, index_pairs in enumerate(zip(*all_indices)):
        print_progress('Analysing fixels', index_i, len(gt_indices), t_init)
        gt_dirs, gt_ini = index_pairs[0]
        gt_peak_i = gt_peak[gt_ini:gt_ini + gt_dirs]
        gt_afd_i = gt_afd[gt_ini:gt_ini + gt_dirs]
        gt_dir_i = gt_dir[gt_ini:gt_ini + gt_dirs, ...]
        m_peak_i = [
            m[m_ini:m_ini + m_dirs]
            for m, (m_dirs, m_ini) in zip(m_peak, index_pairs[1:])
        ]
        m_afd_i = [
            m[m_ini:m_ini + m_dirs]
            for m, (m_dirs, m_ini) in zip(m_afd, index_pairs[1:])
        ]
        m_dir_i = [
            m[m_ini:m_ini + m_dirs, ...]
            for m, (m_dirs, m_ini) in zip(m_dir, index_pairs[1:])
        ]

        indices, angular_error = index_to_gt(gt_dir_i, m_dir_i)
        for m_j, (idx, afd_ij, peak_ij) in enumerate(zip(indices, m_afd_i, m_peak_i)):
            mask = idx >= 0
            gt_afd_in = gt_afd_i[mask]
            gt_afd_out = gt_afd_i[np.logical_not(mask)]
            gt_peak_in = gt_peak_i[mask]
            gt_peak_out = gt_peak_i[np.logical_not(mask)]

            true_idx = idx[mask]
            extra_idx = [
                idx_k for idx_k in range(len(afd_ij))
                if idx_k not in true_idx
            ]
            true_afd = afd_ij[true_idx]
            extra_afd = afd_ij[extra_idx]
            true_peak = peak_ij[true_idx]
            extra_peak = afd_ij[extra_idx]

            m_afd_error = np.abs(true_afd - gt_afd_in).tolist()
            afd_errors[m_j][index_i] = m_afd_error
            # Need to check if tehre are extra errors
            if not np.all(mask):
                missing_afd_errors[m_j][index_i] = gt_afd_out.tolist()
            if np.any(extra_idx):
                extra_afd_errors[m_j][index_i] = extra_afd.tolist()

            m_peak_error = np.abs(true_peak - gt_peak_in).tolist()
            peak_errors[m_j][index_i] = m_peak_error
            # Need to check if tehre are extra errors
            if not np.all(mask):
                missing_peak_errors[m_j][index_i] = gt_peak_out.tolist()
            if np.any(extra_idx):
                extra_peak_errors[m_j][index_i] = extra_peak.tolist()

            angular_ej = angular_error[m_j][mask]
            fixels = len(angular_ej)
            old_count = counts[m_j]
            counts[m_j] += fixels

            angular_errors[m_j] = (
                angular_errors[m_j] * old_count + np.sum(angular_ej)
            ) / counts[m_j] if counts[m_j] != 0 else 0

    afd_tuple = (afd_errors, extra_afd_errors, missing_afd_errors)
    peak_tuple = (peak_errors, extra_peak_errors, missing_peak_errors)

    return angular_errors, afd_tuple, peak_tuple
