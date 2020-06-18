# Adapted from https://blog.evjang.com/2019/07/nf-jax.html
import jax.numpy as np
from jax.experimental.optimizers import adam
from jax import jit, grad, vmap
import jax.random as random
import numpy as onp
from jax.scipy.special import logsumexp
from uniform_flow_model import (
    sample_our_uniform,
    log_prob_our_uniform,
    log_prob_nvp_chain,
    pushforward,
    log_det_J_u,
    init_initial_inv_sigmoid_nvp_chain,
    fused_sample_our_uniform,
)
from utils import (
    uniform_to_cell,
    choose_cells,
    fuse_params,
    unfuse_params,
    init_double_deep_collection,
)
from functools import partial

# config.update("jax_debug_nans", True)

n_test = 200
onp.random.seed(0)

# We need to give a default function, but
# we want to raise an error if this is actually called


def log_p_density(x):
    raise NotImplementedError


# sample_p = sample_three_gaussians

batched_log_p_density = vmap(log_p_density, in_axes=(0,))
# batched_log_p_density = vmap(n_dim_gaussian_grid(5, 2, 0.1))


def log_loss(params, cs, batch):
    return -np.mean(log_prob_nvp_chain(params, cs, log_prob_our_uniform, batch))


def elbo_loss(params, cs, ws, p_density):
    zs = pushforward(params, cs, ws)
    elbo_samples = p_density(zs) + log_det_J_u(params, cs, ws)
    return -np.sum(elbo_samples)


grad_elbo_loss = grad(elbo_loss)


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def compute_base_elbo(ps, cs, log_p, dim, n_samples, tol, rng):
    ws = sample_our_uniform(n_samples, d=dim, rng=rng, tol=1e-4)
    zs = pushforward(ps, cs, ws)
    elbo_samples = log_p(zs) + log_det_J_u(ps, cs, ws)
    return elbo_samples.mean(), elbo_samples.std()


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def compute_importance_sampling(ps, cs, log_p, dim, n_samples, tol, rng):
    ws = sample_our_uniform(n_samples, d=dim, rng=rng, tol=1e-4)
    zs = pushforward(ps, cs, ws)
    importance_sampling_samples = np.exp(log_p(zs) - log_det_J_u(ps, cs, ws))
    return importance_sampling_samples.mean(), importance_sampling_samples.std()


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def compute_uniform_q(ps, cs, log_p, dim, n_samples, tol, rng, num_boxes=1):
    num_cells = 2 ** dim
    deep_ps, deep_cs, lhs = init_double_deep_collection(
        dim, 4, num_boxes, num_cells, rng
    )
    elbos = []
    for (l, h) in lhs:
        batch = sample_our_uniform(n_samples, d=dim, rng=rng, tol=1e-4)
        ws_batch = uniform_to_cell(batch, l, h)
        zs = pushforward(ps, cs, ws_batch)
        elbo_samples = log_p(zs) + log_det_J_u(ps, cs, ws_batch)
        elbo = elbo_samples.mean()
        elbos.append(elbo)
    return logsumexp(elbos) - np.log(len(lhs))


def train_base_flow(
    n_iters=10000,
    n=4,
    objective_fun=batched_log_p_density,
    dim=2,
    tol=1e-4,
    lr=3e-5,
    batch_size=256,
):
    """Trains a flow on objective_fun

    Trains a flow that maps from the uniform density
    to reals, such that we match the density given
    by objective_fun

    Args:
        n_iters: number of gradient steps
        n: number of layers in the flow
        objective_fun: function from (n x dim) to (n x 1),
            giving the log-density of each point in the batch
        dim: dimension of the input
        tol: tolerance in the uniform sampling (i.e.
            we sample from [tol, 1-tol]^d for numerical reasons)
        lr: learning rate in gradient descent
        batch_size: number of points sampled in each minibatch

    Returns:
        ps: a dictionary with the trained flow parameters
        cs: a set of jax objects containing the flow layers
    """
    rng = random.PRNGKey(0)
    ps, cs = init_initial_inv_sigmoid_nvp_chain(
        d=dim, n=n, rng=rng, beta_0s=[1.0, 1.6, -0.5]
    )

    rng, _ = random.split(rng, 2)
    opt_init, opt_update, get_params = adam(lr)
    opt_state = opt_init(ps)

    @partial(jit, static_argnums=(2,))
    def step(i, opt_state, cs, rng):
        params = get_params(opt_state)
        w_batch = sample_our_uniform(batch_size, d=dim, rng=rng, tol=tol)
        g = grad_elbo_loss(params, cs, w_batch, objective_fun)
        rng, _ = random.split(rng, 2)
        return opt_update(i, g, opt_state), g, rng

    best_ps = None
    best_elbo = -np.inf
    for i in range(n_iters):
        opt_state, g, rng = step(i, opt_state, cs, rng)

        if i % 500 == 0:
            ps = get_params(opt_state)
            elbo = compute_base_elbo(ps, cs, objective_fun, dim, 2000, 1e-4, rng)[0]
            # Store the best params as they go
            if elbo > best_elbo:
                best_ps = ps
            print("At iter {}, got elbo: {}".format(i, elbo))

    final_elbo, _ = compute_base_elbo(ps, cs, objective_fun, dim, 2000, 1e-4, rng)
    if best_elbo > elbo or np.isnan(final_elbo):
        ps = best_ps
        if np.abs((best_elbo - final_elbo) / final_elbo) > 0.1 or np.isnan(final_elbo):
            print(
                "Warning: best ELBO was {}, final ELBO was {}".format(
                    best_elbo, final_elbo
                )
            )
    return ps, cs


def do_joint_training(
    input_fun,
    ps=None,
    cs=None,
    dim=2,
    num_cells=16,
    batch_size_inner=128,
    compute_objective_batch_size=128,
    num_boxes_per_step=8,
    lower_step_size=1e-4,
    num_big_steps=5,
    n_base=4,
    tol=1e-4,
    save_frequency=30,
    num_inner_iters=200,
    upper_step_size=3e-4,
    compute_upper_step_size=3e-4,
    compute_objective_inner_iters=3000,
    n_temp=4,
    joint_method="rectified",
    rectified_lambda=0.1,
    burn_in=0,
):
    """Do joint training

    Computes the enhanced ELBO described in the paper

    Args:
        input_fun: batched log density function,
            taking an (n x d) matrix and returning an
            (n x 1) array of log densities
        ps: initial flow parameters if warm starting
        cs: initial flow configs if warm starting
        dim: dimensionality of data
        num_cells: number of cells to partition into
            must be a power of two with the current partitioning
            scheme
        batch_size_inner: number of points sampled in each
            inner batch
        compute_objective_batch_size: number of points sampled
            when computing the elbo
        num_boxes_per_step: number of cells sampled in each
            `minibatch` of cells
        lower_step_size: [[]]
        num_big_steps: number of outer steps taken before terminating
        n_base: number of layers in the base flow
        tol: tolerance in the uniform sampling for the flows
        save_frequency: how many steps between each save of the ELBO
        num_inner_iters: number of inner iterations between each inner
            step
        upper_step_size: step size in the elbo computation phase
        compute_upper_step_size: step size in the upper elbo computation phase
        n_temp: number of layers in the temporary flows instantiated
        joint_method: the method taken to do the joint training
        rectified_lambda: the mixing coefficient for the global elbo
            in the rectified joint method case. Possible values are
            'rectified', 'mean_elbo', or 'exp_elbo'

    Returns:
        ELBO of input_fun, computed via the method in the paper.

    """
    compute_objective_num_box = num_boxes_per_step
    opt_init_lower, opt_update_lower, get_params_lower = adam(lower_step_size)
    opt_init_upper, opt_update_upper, get_params_upper = adam(upper_step_size)
    opt_init_compute_upper, opt_update_compute_upper, get_params_compute_upper = adam(
        compute_upper_step_size
    )

    rng = random.PRNGKey(0)
    if ps is None:
        ps, cs = init_initial_inv_sigmoid_nvp_chain(d=dim, n=n_base, rng=rng)
    (
        static_deep_ps_collection,
        static_deep_cs_collection,
        lhs,
    ) = init_double_deep_collection(dim, n_temp, num_boxes_per_step, num_cells, rng)

    def shallow_nelbo(our_ps, batch):
        batch = batch.reshape((-1, dim))
        log_det_J_layer_2 = log_det_J_u(our_ps, cs, batch)
        p_densities = input_fun(pushforward(our_ps, cs, batch))
        return -np.mean(log_det_J_layer_2 + p_densities)

    def deep_nelbo(deep_ps, our_ps, batch, lhs):
        ws_batch = pushforward(deep_ps, static_deep_cs_collection[0], batch)
        ws_batch = uniform_to_cell(ws_batch, lhs[0], lhs[1])
        log_det_J_layer_1 = log_det_J_u(deep_ps, static_deep_cs_collection[0], batch)
        log_det_J_layer_2 = log_det_J_u(our_ps, cs, ws_batch)
        p_densities = input_fun(pushforward(our_ps, cs, ws_batch))
        return -np.mean(log_det_J_layer_1 + log_det_J_layer_2 + p_densities)

    vmapped_nelbo = vmap(deep_nelbo, in_axes=(0, None, 0, 0))

    def batch_loss(fused_ps, ps_1, fused_batches, fused_lhs):
        losses = vmapped_nelbo(fused_ps, ps_1, fused_batches, fused_lhs)
        return -logsumexp(-losses) + np.log(fused_batches.shape[0])

    def average_nelbo(fused_ps, ps_1, fused_batches, fused_lhs, _lambda=0.1):
        losses = vmapped_nelbo(fused_ps, ps_1, fused_batches, fused_lhs)
        return np.mean(losses) + np.std(losses) * _lambda

    def average_rectified_nelbo(
        fused_ps, ps_1, fused_batches, fused_lhs, lam=rectified_lambda
    ):
        cell_losses = vmapped_nelbo(fused_ps, ps_1, fused_batches, fused_lhs)
        main_loss = shallow_nelbo(ps_1, fused_batches)
        return np.mean(cell_losses) * (1 - lam) + np.mean(main_loss) * lam

    grad_upper = grad(batch_loss, argnums=(0,))
    grad_lower = grad(batch_loss, argnums=(1,))
    grad_upper_mean_elbo = grad(average_nelbo, argnums=(0,))
    grad_joint_mean_elbo = grad(average_nelbo, argnums=(0, 1))
    grad_upper_mean_rectified_elbo = grad(average_rectified_nelbo, argnums=(0, 1))

    @jit
    def lower_step(i, opt_state, fused_params, rng, fused_lhs):
        our_ps = get_params_lower(opt_state)
        fused_batches = fused_sample_our_uniform(
            batch_size_inner, num_boxes_per_step, dim, rng, tol=tol
        )
        g = grad_lower(fused_params, our_ps, fused_batches, fused_lhs)[0]
        rng, _ = random.split(rng, 2)
        return opt_update_lower(i, g, opt_state), rng

    @jit
    def upper_step(i, opt_state, our_ps, rng, fused_lhs):
        fused_ps = get_params_upper(opt_state)
        fused_batches = fused_sample_our_uniform(
            batch_size_inner, num_boxes_per_step, dim, rng, tol=tol
        )
        g = grad_upper(fused_ps, our_ps, fused_batches, fused_lhs)[0]
        rng, _ = random.split(rng, 2)
        return opt_update_upper(i, g, opt_state), rng

    @jit
    def joint_step(i, opt_state, rng, fused_lhs):
        fused_ps, our_ps = get_params_upper(opt_state)
        fused_batches = fused_sample_our_uniform(
            batch_size_inner, num_boxes_per_step, dim, rng, tol=tol
        )
        g = grad_joint_mean_elbo(fused_ps, our_ps, fused_batches, fused_lhs)
        rng, _ = random.split(rng, 2)
        return opt_update_upper(i, g, opt_state), rng

    @jit
    def upper_compute_step(i, opt_state, our_ps, rng, fused_lhs):
        """Separate function so we can use a lower learning rate"""
        fused_ps = get_params_compute_upper(opt_state)
        fused_batches = fused_sample_our_uniform(
            compute_objective_batch_size, compute_objective_num_box, dim, rng, tol=tol
        )
        g = grad_upper(fused_ps, our_ps, fused_batches, fused_lhs)[0]
        rng, _ = random.split(rng, 2)
        return opt_update_compute_upper(i, g, opt_state), rng

    @jit
    def upper_average_compute_step(i, opt_state, our_ps, rng, fused_lhs):
        """Separate function so we can use a lower learning rate"""
        fused_ps = get_params_compute_upper(opt_state)
        fused_batches = fused_sample_our_uniform(
            batch_size_inner, compute_objective_num_box, dim, rng, tol=tol
        )
        g = grad_upper_mean_elbo(fused_ps, our_ps, fused_batches, fused_lhs)[0]
        rng, _ = random.split(rng, 2)
        return opt_update_compute_upper(i, g, opt_state), rng, g

    @jit
    def upper_average_rectfied_step(i, opt_state, rng, fused_lhs):
        """Separate function so we can use a lower learning rate"""
        fused_ps, _ps = get_params_compute_upper(opt_state)
        fused_batches = fused_sample_our_uniform(
            batch_size_inner, compute_objective_num_box, dim, rng, tol=tol
        )
        g = grad_upper_mean_rectified_elbo(fused_ps, _ps, fused_batches, fused_lhs)
        rng, _ = random.split(rng, 2)
        return opt_update_upper(i, g, opt_state), rng

    def compute_objective(fixed_ps, num_iters, num_boxes, num_cells, rng):
        deep_ps, deep_cs, lhs = init_double_deep_collection(
            dim, n_temp, num_boxes, num_cells, rng
        )
        fused_ps = fuse_params(deep_ps)
        fused_lhs = fuse_params(lhs)
        opt_state = opt_init_compute_upper(fused_ps)
        best_fused_elbos = onp.ones(num_boxes) * -np.inf
        best_unfused_ps = deep_ps
        for i in range(num_iters):
            opt_state, rng = upper_compute_step(i, opt_state, fixed_ps, rng, fused_lhs)
            if i % save_frequency == 0:
                fused_ps = get_params_compute_upper(opt_state)
                fused_batches = fused_sample_our_uniform(
                    5 * compute_objective_batch_size, num_boxes, dim, rng, tol=tol
                )
                new_objective_losses = vmapped_nelbo(
                    fused_ps, fixed_ps, fused_batches, fused_lhs
                )
                # print(f"Test Loss is {-new_objective_losses}")
                # print(f"Test Total Loss is {logsumexp(-new_objective_losses) - np.log(num_boxes)}")
                for k in range(num_boxes):
                    if -new_objective_losses[k] > best_fused_elbos[k]:
                        best_fused_elbos[k] = -new_objective_losses[k]
                        best_unfused_ps[k] = unfuse_params(fused_ps)[k]
        fused_ps = fuse_params(best_unfused_ps)
        fused_batches = fused_sample_our_uniform(
            5 * compute_objective_batch_size, num_boxes, dim, rng, tol=tol
        )
        new_objective_loss = batch_loss(fused_ps, fixed_ps, fused_batches, fused_lhs)
        return -new_objective_loss

    def joint_procedure(rng, in_ps):
        elbo_estimates = []
        if joint_method == "pure_joint":
            opt_state_lower = opt_init_lower(in_ps)
        burn_in_counter = 0
        for j in range(num_big_steps):
            burn_in_counter += 1
            lhs, rng = choose_cells(dim, num_boxes_per_step, num_cells, rng)
            fused_ps = fuse_params(static_deep_ps_collection)
            fused_lhs = fuse_params(lhs)

            if joint_method == "pure_joint":
                opt_state_upper = opt_init_upper(fused_ps)
            else:
                # Use a joint training method
                opt_state_upper = opt_init_upper((fused_ps, in_ps))
            best_elbo = -np.inf
            best_fused_ps = None
            for k in range(num_inner_iters):
                if joint_method == "pure_joint":
                    opt_state_upper, rng = upper_step(
                        k, opt_state_upper, in_ps, rng, fused_lhs
                    )
                elif joint_method == "rectified":
                    opt_state_upper, rng = upper_average_rectfied_step(
                        k, opt_state_upper, rng, fused_lhs
                    )
                else:
                    assert joint_method == "joint_naive"
                    opt_state_upper, rng = joint_step(
                        k, opt_state_upper, rng, fused_lhs
                    )
                if k % save_frequency == 0:
                    fused_batches = fused_sample_our_uniform(
                        compute_objective_batch_size,
                        num_boxes_per_step,
                        dim,
                        rng,
                        tol=tol,
                    )
                    if joint_method == "pure_joint":
                        fused_ps = get_params_upper(opt_state_upper)
                    else:
                        fused_ps, in_ps = get_params_upper(opt_state_upper)
                    new_objective_loss = batch_loss(
                        fused_ps, in_ps, fused_batches, fused_lhs
                    )
                    # print("ELBO running estimate: {}".format(-new_objective_loss))

                    if -new_objective_loss > best_elbo:
                        best_elbo = -new_objective_loss
                        best_fused_ps = fused_ps

            print("At iter {}".format(j))
            fused_ps = best_fused_ps
            if joint_method == "pure_joint":
                opt_state_lower, rng = lower_step(
                    j, opt_state_lower, fused_ps, rng, fused_lhs
                )
                in_ps = get_params_lower(opt_state_lower)
            elbo_estimate = compute_objective(
                in_ps,
                compute_objective_inner_iters,
                compute_objective_num_box,
                num_cells,
                rng,
            )
            if burn_in_counter > burn_in:
                elbo_estimates.append(elbo_estimate)
            print(elbo_estimate)

            # with open(filestring, 'rb') as f:
            # new_ps = pickle.load(f)
            # plot_save_2_lines_density(in_ps, cs, new_ps, cs, input_fun)

            print("ELBO real estimate: {}".format(elbo_estimate))
        elbo_estimates = np.concatenate(
            [x.reshape((1,)) for x in elbo_estimates if not np.isnan(x)]
        )
        return elbo_estimates

    # for i in range(num_big_steps):
    #     elbo_samples, ps, rng = new_outer_step(i, rng, ps)
    # return ps

    estimates = joint_procedure(rng, ps)
    return estimates.mean(), estimates.std()
