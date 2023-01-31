import numpy as np
import time
from copy import deepcopy
from utils import print_progress


def index_to_gt(gt, others):
    """
    Function to find the matching vectors (lowest angular error) between
    a list of gold standard vectors and a list of lists of predicted /
    downsampled vectors.
    :param gt: Gold standard unit vectors for a given fixel.
    :param others: List of unit vectors for a given fixel and method.
    :return: Matching indices between the gold standard vectors and methods and
     a list of the angular error for that fixel.
    """
    # Init
    n_gt = len(gt)
    indices = []
    errors = []

    # Loop for all the methods.
    # We always assume others is a list of methods, and each 'method' contains
    # a list of directions (unit vectors).
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
    target_indices, target_peak, target_afd, target_dir,
    source_indices, source_peak, source_afd, source_dir,
):
    """
    Function to do the whole fixel-based comparison for a gold standard set
    of fixels and a list of predicted ones.
    :param target_indices: Gold standard fixel indices (to use with afd, peak
     and dir).
    :param target_peak: Gold standard peaks.
    :param target_afd: Gold standard apparent fibre density values.
    :param target_dir: Gold standard fixel directions.
    :param source_indices: Predicted / downsampled fixel indices (to use with
     afd, peak and dir).
    :param source_peak: Predicted / downsampled peaks.
    :param source_afd: Predicted / downsampled apparent fibre density values.
    :param source_dir: Predicted / downsampled fixel directions.
    :return: Angular errors per method, a tuple with the peak errors and a
     tuple with the afd errors.
    """
    # Init
    # In order to increase speed, we save storage for all the results here.
    # Extending lists for each fixel has a costy penalty.
    angular_errors = [0. for _ in source_indices]
    afd_errors = [
        [
            np.zeros(min(gt_i, m_i))
            for (gt_i, _), (m_i, _) in zip(target_indices, m_index)
        ]
        for m_index in source_indices
    ]
    missing_afd_errors = [
        [
            np.zeros(gt_i - m_i) if gt_i > m_i else np.zeros(1)
            for (gt_i, _), (m_i, _) in zip(target_indices, m_index)
        ]
        for m_index in source_indices
    ]
    extra_afd_errors = [
        [
            np.zeros(m_i - gt_i) if gt_i < m_i else np.zeros(1)
            for (gt_i, _), (m_i, _) in zip(target_indices, m_index)
        ]
        for m_index in source_indices
    ]
    peak_errors = deepcopy(afd_errors)
    missing_peak_errors = deepcopy(missing_afd_errors)
    extra_peak_errors = deepcopy(extra_afd_errors)
    counts = np.zeros(len(source_indices), dtype=np.float64)
    all_indices = tuple([target_indices] + source_indices)
    t_init = time.time()

    # Main fixel loop
    # Due to the data structure, we need to loop through the indices first
    # to be able to access matching fixels.
    for index_i, index_pairs in enumerate(zip(*all_indices)):
        print_progress(
            'Analysing fixels', index_i, len(target_indices), t_init
        )
        gt_dirs, gt_ini = index_pairs[0]
        gt_peak_i = target_peak[gt_ini:gt_ini + gt_dirs]
        gt_afd_i = target_afd[gt_ini:gt_ini + gt_dirs]
        gt_dir_i = target_dir[gt_ini:gt_ini + gt_dirs, ...]
        m_peak_i = [
            m[m_ini:m_ini + m_dirs]
            for m, (m_dirs, m_ini) in zip(source_peak, index_pairs[1:])
        ]
        m_afd_i = [
            m[m_ini:m_ini + m_dirs]
            for m, (m_dirs, m_ini) in zip(source_afd, index_pairs[1:])
        ]
        m_dir_i = [
            m[m_ini:m_ini + m_dirs, ...]
            for m, (m_dirs, m_ini) in zip(source_dir, index_pairs[1:])
        ]

        # This next function is key to find matching fixels.
        # indices will only contain at most all the predicted / downsampled
        # indices to fixels matching a gold standard fixel (without
        # repetitions). If there are more fixels on the gold standard, the
        # extra gold standard indices will be set to -1. If there are extra
        # fixels on the predicted / downsampled list, the extra fixels will
        # not be referenced in indices.
        indices, angular_error = index_to_gt(gt_dir_i, m_dir_i)
        for m_j, (idx, afd_ij, peak_ij) in enumerate(
                zip(indices, m_afd_i, m_peak_i)
        ):
            # For the peak and apparent fibre density error, we assume only the
            # matched fixels can be counted. On one hand, this leads to an
            # underestimation of the error (as missing fixels or extra ones are
            # not taken into account). On the other hand it gives a better view
            # on the fixels that were found. However, we also include the
            # missing or additional fixes in their own lists to account for
            # this shortcoming.
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

            afd_errors[m_j][index_i] = np.abs(true_afd - gt_afd_in)
            peak_errors[m_j][index_i] = np.abs(true_peak - gt_peak_in)

            # Need to check if there are extra errors
            if not np.all(mask):
                missing_afd_errors[m_j][index_i] = gt_afd_out.tolist()
                missing_peak_errors[m_j][index_i] = gt_peak_out.tolist()
            if np.any(extra_idx):
                extra_afd_errors[m_j][index_i] = extra_afd.tolist()
                extra_peak_errors[m_j][index_i] = extra_peak.tolist()

            # Cumulative mean angular error. We keep a moving count variable
            # to adapt the mean as new fixels are accounted for.
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
