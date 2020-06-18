import jax.numpy as np
import jax.random as random
import jax.ops as ops
from jax.scipy.special import logsumexp


def sample_7_dim_proposal(n, rng):
    sigma = 0.1
    a_1s = random.normal(rng, (n, 1)) * sigma
    rng, _ = random.split(rng, 2)
    a_2 = (random.normal(rng, (n, 1)) * sigma + 1).reshape((-1, 1))
    rng, _ = random.split(rng, 2)
    b_2 = (random.normal(rng, (n, 1)) * sigma + 0).reshape((-1, 1))
    one_zeros = np.concatenate((a_2, b_2), axis=1)
    rng, _ = random.split(rng, 2)
    a_3 = (random.normal(rng, (n, 1)) * sigma + -1).reshape((-1, 1))

    rng, _ = random.split(rng, 2)
    b_3 = (random.normal(rng, (n, 1)) * sigma + -1).reshape((-1, 1))
    mones_mones = np.concatenate((a_3, b_3), axis=1)
    rng, _ = random.split(rng, 2)
    a_4 = (random.normal(rng, (n, 1)) * sigma + 0.5).reshape((-1, 1))
    rng, _ = random.split(rng, 2)
    b_4 = (random.normal(rng, (n, 1)) * sigma + -2).reshape((-1, 1))
    halfs_min_2s = np.concatenate((a_4, b_4), axis=1)

    first_sixth = np.concatenate(
        (one_zeros[: n // 6], mones_mones[: n // 6], halfs_min_2s[: n // 6]), axis=1
    )
    second_sixth = np.concatenate(
        (
            one_zeros[n // 6 : 2 * n // 6],
            halfs_min_2s[n // 6 : 2 * n // 6],
            mones_mones[n // 6 : 2 * n // 6],
        ),
        axis=1,
    )
    third_sixth = np.concatenate(
        (
            mones_mones[2 * n // 6 : 3 * n // 6],
            halfs_min_2s[2 * n // 6 : 3 * n // 6],
            one_zeros[2 * n // 6 : 3 * n // 6],
        ),
        axis=1,
    )
    fourth_sixth = np.concatenate(
        (
            mones_mones[3 * n // 6 : 4 * n // 6],
            one_zeros[3 * n // 6 : 4 * n // 6],
            halfs_min_2s[3 * n // 6 : 4 * n // 6],
        ),
        axis=1,
    )
    fifth_sixth = np.concatenate(
        (
            halfs_min_2s[4 * n // 6 : 5 * n // 6],
            one_zeros[4 * n // 6 : 5 * n // 6],
            mones_mones[4 * n // 6 : 5 * n // 6],
        ),
        axis=1,
    )
    sixth_sixth = np.concatenate(
        (
            halfs_min_2s[5 * n // 6 : 6 * n // 6],
            mones_mones[5 * n // 6 : 6 * n // 6],
            one_zeros[5 * n // 6 : 6 * n // 6],
        ),
        axis=1,
    )
    all_paired_samples = np.concatenate(
        (
            first_sixth,
            second_sixth,
            third_sixth,
            fourth_sixth,
            fifth_sixth,
            sixth_sixth,
        ),
        axis=0,
    )
    samples = np.concatenate((a_1s, all_paired_samples), axis=1)
    mode_1_means = np.array([0, 1, 0, -1, -1, 0.5, -2])
    mode_2_means = np.array([0, 1, 0, 0.5, -2, -1, -1])
    mode_3_means = np.array([0, -1, -1, 0.5, -2, 1, 0])
    mode_4_means = np.array([0, -1, -1, 1, 0, 0.5, -2])
    mode_5_means = np.array([0, 0.5, -2, -1, -1, 1, 0])
    mode_6_means = np.array([0, 0.5, -2, 1, 0, -1, -1])

    new_samples = np.zeros((n, 7))
    new_samples = ops.index_update(new_samples, ops.index[:, :2], samples[:, :2])
    new_samples = ops.index_update(new_samples, ops.index[:, 2], samples[:, 3])
    new_samples = ops.index_update(new_samples, ops.index[:, 3], samples[:, 5])
    new_samples = ops.index_update(new_samples, ops.index[:, 4], samples[:, 2])
    new_samples = ops.index_update(new_samples, ops.index[:, 5], samples[:, 4])
    new_samples = ops.index_update(new_samples, ops.index[:, 6], samples[:, 6])
    log_densities_prefactor = -0.5 * np.log(2 * np.pi * sigma ** 2) - np.log(6)
    log_densities_distance_factor = logsumexp(
        np.array(
            [
                -0.5 * np.sum(((samples - mode_1_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_2_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_3_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_4_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_5_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_6_means) ** 2) / (sigma ** 2), axis=1),
            ]
        )
    )

    return new_samples, log_densities_distance_factor + log_densities_prefactor


def sample_6_dim_proposal(n, rng):
    sigma = 0.1
    a_2 = (random.normal(rng, (n, 1)) * sigma + 1).reshape((-1, 1))
    rng, _ = random.split(rng, 2)
    b_2 = (random.normal(rng, (n, 1)) * sigma + 0).reshape((-1, 1))
    one_zeros = np.concatenate((a_2, b_2), axis=1)
    rng, _ = random.split(rng, 2)
    a_3 = (random.normal(rng, (n, 1)) * sigma + -1).reshape((-1, 1))
    rng, _ = random.split(rng, 2)
    b_3 = (random.normal(rng, (n, 1)) * sigma + -1).reshape((-1, 1))
    mones_mones = np.concatenate((a_3, b_3), axis=1)
    rng, _ = random.split(rng, 2)
    a_4 = (random.normal(rng, (n, 1)) * sigma + 0.5).reshape((-1, 1))
    rng, _ = random.split(rng, 2)
    b_4 = (random.normal(rng, (n, 1)) * sigma + -2).reshape((-1, 1))
    halfs_min_2s = np.concatenate((a_4, b_4), axis=1)

    first_sixth = np.concatenate(
        (one_zeros[: n // 6], mones_mones[: n // 6], halfs_min_2s[: n // 6]), axis=1,
    )
    second_sixth = np.concatenate(
        (
            one_zeros[n // 6 : 2 * n // 6],
            halfs_min_2s[n // 6 : 2 * n // 6],
            mones_mones[n // 6 : 2 * n // 6],
        ),
        axis=1,
    )
    third_sixth = np.concatenate(
        (
            mones_mones[2 * n // 6 : 3 * n // 6],
            halfs_min_2s[2 * n // 6 : 3 * n // 6],
            one_zeros[2 * n // 6 : 3 * n // 6],
        ),
        axis=1,
    )
    fourth_sixth = np.concatenate(
        (
            mones_mones[3 * n // 6 : 4 * n // 6],
            one_zeros[3 * n // 6 : 4 * n // 6],
            halfs_min_2s[3 * n // 6 : 4 * n // 6],
        ),
        axis=1,
    )
    fifth_sixth = np.concatenate(
        (
            halfs_min_2s[4 * n // 6 : 5 * n // 6],
            one_zeros[4 * n // 6 : 5 * n // 6],
            mones_mones[4 * n // 6 : 5 * n // 6],
        ),
        axis=1,
    )
    sixth_sixth = np.concatenate(
        (
            halfs_min_2s[5 * n // 6 : 6 * n // 6],
            mones_mones[5 * n // 6 : 6 * n // 6],
            one_zeros[5 * n // 6 : 6 * n // 6],
        ),
        axis=1,
    )
    all_paired_samples = np.concatenate(
        (
            first_sixth,
            second_sixth,
            third_sixth,
            fourth_sixth,
            fifth_sixth,
            sixth_sixth,
        ),
        axis=0,
    )
    samples = all_paired_samples
    mode_1_means = np.array([1, 0, -1, -1, 0.5, -2])
    mode_2_means = np.array([1, 0, 0.5, -2, -1, -1])
    mode_3_means = np.array([-1, -1, 0.5, -2, 1, 0])
    mode_4_means = np.array([-1, -1, 1, 0, 0.5, -2])
    mode_5_means = np.array([0.5, -2, -1, -1, 1, 0])
    mode_6_means = np.array([0.5, -2, 1, 0, -1, -1])

    new_samples = np.zeros((n, 6))
    new_samples = ops.index_update(new_samples, ops.index[:, 0], samples[:, 0])
    new_samples = ops.index_update(new_samples, ops.index[:, 1], samples[:, 2])
    new_samples = ops.index_update(new_samples, ops.index[:, 2], samples[:, 4])
    new_samples = ops.index_update(new_samples, ops.index[:, 3], samples[:, 1])
    new_samples = ops.index_update(new_samples, ops.index[:, 4], samples[:, 3])
    new_samples = ops.index_update(new_samples, ops.index[:, 5], samples[:, 5])
    log_densities_prefactor = -0.5 * np.log(2 * np.pi * sigma ** 2) - np.log(6)
    log_densities_distance_factor = logsumexp(
        np.array(
            [
                -0.5 * np.sum(((samples - mode_1_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_2_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_3_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_4_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_5_means) ** 2) / (sigma ** 2), axis=1),
                -0.5 * np.sum(((samples - mode_6_means) ** 2) / (sigma ** 2), axis=1),
            ]
        )
    )

    return new_samples, log_densities_distance_factor + log_densities_prefactor
