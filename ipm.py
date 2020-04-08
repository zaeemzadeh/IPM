import numpy as np
import irlb


def ipm(data, n=1, selected_data=None):
    # data: list of data to be selected
    # n:    number of samples to be selected
    # selected_data: list of already selected data
    if selected_data is None:
        selected_data = []

    pooled_idx = []
    while len(pooled_idx) < n:
        new_idx = ipm_add_sample(selected_data, data, pooled_idx)
        pooled_idx.append(int(new_idx))
    return pooled_idx


def ipm_add_sample(train, pool, pooled_idx):
    candidate_samples = range(0, len(pool))
    set_idx = [int(idx) for idx in pooled_idx]

    # generating the matrix of already selected samples
    A_train = [np.ravel(t) for t in train]
    A_train.extend([np.ravel(pool[i]) for i in set_idx])
    A_s_mat = np.array(A_train).transpose()
    if len(A_s_mat.shape) == 1:
        A_s_mat = A_s_mat.reshape((-1, 1))

    # generating the matrix of data
    A_mat = np.array([np.ravel(t) for t in pool]).transpose()
    if len(A_mat.shape) == 1:
        A_mat = A_mat.reshape((-1, 1))

    # projecting onto the nullspace of the selected data
    if len(A_s_mat) == 0:
        A_proj = A_mat
    else:
        Proj = np.matmul(A_s_mat, np.linalg.pinv(A_s_mat))
        A_proj = A_mat - np.matmul(Proj, A_mat)

    # calculating the first singular vector
    first_eig_vec = irlb.irlb(A_proj, 2)[0][:, 0]

    # calculating the correlations
    correlation = np.zeros(len(pool))
    for m in candidate_samples:

        if m in pooled_idx:
            correlation[m] = 0
            continue

        correlation[m] = np.abs(np.inner(A_mat[:, m], first_eig_vec))
        correlation[m] /= np.linalg.norm(np.squeeze(A_mat[:, m]))

    # finding the best sample
    sorted_idx = np.argsort(correlation)

    return sorted_idx[-1]