import numpy as np
from sklearn import metrics
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table_from_bvals_bvecs


def kenStone(X, k, random=False, metric='euclidean'):
    # Number of samples
    n = len(X)

    # Pair-wise distance matrix
    dist = metrics.pairwise_distances(X, metric=metric, n_jobs=-1)

    # Get the first samples
    if random:
        i0 = np.random.randint(n)
        selected = set([i0])
        k -= 1
    else:
        i0, i1 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
        selected = set([i0, i1])
        k -= 2

    # Iterate to find the rest
    minj = i0
    while k > 0 and len(selected) < n:
        mindist = 0.0
        for j in range(n):
            if j not in selected:
                mindistj = min([dist[j][i] for i in selected])
                if mindistj > mindist:
                    minj = j
                    mindist = mindistj
        selected.add(minj)
        k -= 1
    return list(selected)


def extract_single_shell(
        bvals, bvecs, extract_bval=1000, extract_range=20, directions=32,
        sample=True
):
    # Load original dwi and fslgrad.
    if isinstance(bvals, str) and isinstance(bvecs, str):
        bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    gtab = gradient_table_from_bvals_bvecs(bvals, bvecs)

    # Obtain the index for the selected bvalue.
    idx = np.logical_and(
        (extract_bval - extract_range) <= gtab.bvals,
        gtab.bvals <= (extract_bval + extract_range)
    )
    idx_0 = np.logical_and(
        (0 - extract_range) <= gtab.bvals,
        gtab.bvals <= (0 + extract_range)
    )

    # Get the extracted bvals and bvecs
    new_bvals = gtab.bvals[idx]
    new_bvecs = gtab.bvecs[idx]

    normalized_bvecs = new_bvecs / np.linalg.norm(new_bvecs, axis=1, keepdims=True)
    bvecs_input = normalized_bvecs.copy()
    bvecs_input[bvecs_input[:, -1] < 0, :] = - bvecs_input[bvecs_input[:, -1] < 0, :]

    # Use function to compute index
    if sample:
        lr_index = kenStone(bvecs_input, directions)

        # Extract low gradient direction resolution data
        lr_bvecs = new_bvecs[lr_index]
        lr_bvals = new_bvals[lr_index]

        lr_index = np.where(idx)[0][lr_index]
    else:
        lr_bvecs = new_bvecs
        lr_bvals = new_bvals

        lr_index = np.where(idx)[0]

    return lr_bvecs, lr_bvals, lr_index, np.where(idx_0)[0]
