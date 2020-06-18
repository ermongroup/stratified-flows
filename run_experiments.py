# Adapted from https://blog.evjang.com/2019/07/nf-jax.html
import jax.numpy as np
from jax import vmap, jit
from jax.random import PRNGKey
import jax.random as random
import jax.ops as ops
import numpy as onp
from jax.scipy.special import logsumexp
from uniform_flow_model import sample_our_uniform
from bayes import log_integrand
from training import (
    train_base_flow,
    compute_base_elbo,
    do_joint_training,
)
from sampling import sample_7_dim_proposal, sample_6_dim_proposal


def do_bayesian_test():
    """The idea with this one is to compute P(a_1|b_1=2,y,x) = P(a_1,b_1=2,y,x)/P(b_1=2,y,x)"""
    rng = PRNGKey(0)
    tol = 1e-6
    # Now going to try this in eight dimensions
    x = np.load("./data/x.npy")
    y = np.load("./data/y.npy")
    n_side = 20

    full_variational_approach_results = onp.zeros((n_side,))
    small_variational_approach_results = onp.zeros((n_side,))
    our_approach_results = onp.zeros((n_side,))

    sigma = 0.1

    def partial_objective_fun(partial_params):
        """We pass in a 7-dim vector and fix b_1=2 to make an 8-dim vector"""
        big_params = np.zeros(8)
        b_1 = 2
        big_params = ops.index_update(big_params, ops.index[:4], partial_params[:4])
        big_params = ops.index_update(big_params, ops.index[4], b_1)
        big_params = ops.index_update(big_params, ops.index[5:], partial_params[4:])
        return log_integrand(y, x, big_params, sigma)

    print("With 8...")
    full_ps, full_cs = train_base_flow(
        10000, 8, vmap(partial_objective_fun), dim=7, tol=tol, lr=3e-5
    )
    variational_denominator, std = compute_base_elbo(
        full_ps, full_cs, vmap(partial_objective_fun), 7, 2000, tol, rng
    )
    print(variational_denominator, std)
    print("With 4...")
    full_ps, full_cs = train_base_flow(
        10000, 4, vmap(partial_objective_fun), dim=7, tol=1e-4, lr=3e-5
    )
    variational_denominator, std = compute_base_elbo(
        full_ps, full_cs, vmap(partial_objective_fun), 7, 2000, tol, rng
    )
    print(variational_denominator, std)
    rng, _ = random.split(rng, 2)
    joint_denominator, std = do_joint_training(
        vmap(partial_objective_fun),
        dim=7,
        num_cells=128,
        batch_size_inner=128,
        num_boxes_per_step=12,
        num_big_steps=8,
        num_inner_iters=600,
        upper_step_size=3e-4,
        compute_upper_step_size=3e-5,
        compute_objective_inner_iters=1000,
        joint_method="rectified",
        rectified_lambda=0.1,
        save_frequency=100,
        burn_in=4,
    )
    print(joint_denominator, std)

    ests = []
    num_trials = 40
    sample_lim = 4
    for n in range(num_trials):
        samples = (sample_our_uniform(1500000, 7, rng) - 0.5) * sample_lim
        rng, _ = random.split(rng, 2)
        vals = jit(vmap(partial_objective_fun))(samples)
        log_mean_f_val = logsumexp(vals) - np.log(len(vals))
        sampled_denominator = log_mean_f_val + 7 * np.log(sample_lim)
        ests.append(sampled_denominator)
    ests = np.array(ests)
    print(f"Uniform sampling gives {ests.mean()}, {ests.std()}")

    num_trials = 10
    l_sums = []
    for j in range(num_trials):
        proposal_samples, log_densities = sample_7_dim_proposal(1000000, rng)
        rng, _ = random.split(rng, 2)
        log_summands = (
            jit(vmap(partial_objective_fun))(proposal_samples) - log_densities
        )
        l_sums.append(log_summands)
    log_summands = np.concatenate(l_sums)
    estimate = logsumexp(log_summands) - np.log(len(proposal_samples) * num_trials)
    print(f"Importance Sampled Denominator: {estimate}")

    a_idx = 0
    a_1 = 0

    def partial_objective_fun(partial_params):
        b_1 = 2
        big_params = np.zeros(8)
        big_params = ops.index_update(big_params, ops.index[1:4], partial_params[:3])
        big_params = ops.index_update(big_params, ops.index[5:], partial_params[3:])
        big_params = ops.index_update(big_params, ops.index[0], a_1)
        big_params = ops.index_update(big_params, ops.index[4], b_1)
        return log_integrand(y, x, big_params, sigma)

    print("With 4..")
    ps, cs = train_base_flow(10000, 12, vmap(partial_objective_fun), 6)
    variational_numerator, std = compute_base_elbo(
        ps, cs, vmap(partial_objective_fun), 6, 2000, tol, rng
    )
    rng, _ = random.split(rng, 2)
    print("Variational Numerator: {} pm {}".format(variational_numerator, std))
    small_variational_approach_results[a_idx] = variational_numerator

    print("With 8..")
    ps, cs = train_base_flow(10000, 16, vmap(partial_objective_fun), 6)
    variational_numerator, std = compute_base_elbo(
        ps, cs, vmap(partial_objective_fun), 6, 2000, tol, rng
    )
    rng, _ = random.split(rng, 2)
    print("Variational Numerator: {} pm {}".format(variational_numerator, std))
    full_variational_approach_results[a_idx] = variational_numerator

    joint_numerator, std = do_joint_training(
        vmap(partial_objective_fun),
        dim=6,
        num_cells=64,
        batch_size_inner=128,
        num_boxes_per_step=12,
        num_big_steps=8,
        num_inner_iters=600,
        upper_step_size=3e-4,
        compute_upper_step_size=3e-5,
        compute_objective_inner_iters=1000,
        joint_method="rectified",
        rectified_lambda=0.1,
        save_frequency=100,
        burn_in=4,
    )
    our_approach_results[a_idx] = joint_numerator
    print("Joint Numerator: {} pm {}".format(joint_numerator, std))

    ests = []
    num_trials = 40
    sample_lim = 4
    for n in range(num_trials):
        samples = (sample_our_uniform(1500000, 6, rng) - 0.5) * sample_lim
        rng, _ = random.split(rng, 2)
        vals = jit(vmap(partial_objective_fun))(samples)
        log_mean_f_val = logsumexp(vals) - np.log(len(vals))
        sampled_denominator = log_mean_f_val + 6 * np.log(sample_lim)
        ests.append(sampled_denominator)
    ests = np.array(ests)
    print(f"Uniform sampling gives {ests.mean()}, {ests.std()}")

    num_trials = 10
    l_sums = []
    for j in range(num_trials):
        proposal_samples, log_densities = sample_6_dim_proposal(1000000, rng)
        rng, _ = random.split(rng, 2)
        log_summands = (
            jit(vmap(partial_objective_fun))(proposal_samples) - log_densities
        )
        l_sums.append(log_summands)
    log_summands = np.concatenate(l_sums)
    estimate = logsumexp(log_summands) - np.log(len(proposal_samples) * num_trials)
    print(f"Importance Sampled Numerator: {estimate}")


do_bayesian_test()
