import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
import jax.random as random
import jax.ops as ops
from functools import wraps
from time import time
from uniform_flow_model import (
    init_final_sigmoid_nvp_chain,
    init_identity_nvp_chain,
    sample_our_uniform,
    pushforward,
    init_double_sigmoid_nvp_chain,
)
from jax.tree_util import tree_multimap, tree_flatten, tree_map


def sample_to_cell(ps, cs, n, low_corner, high_corner, rng):
    """Samples from the base distribution and pushes them through
    the flow, then maps them into the specified cell"""
    unit_cell_points = sample_our_uniform(N=n, d=2, rng=rng)
    low_corner_array = np.array(low_corner)
    high_corner_array = np.array(high_corner)
    side_lengths = high_corner_array - low_corner_array
    shifted_cell_points = (unit_cell_points * side_lengths) + low_corner_array
    return shifted_cell_points


def pushforward_to_cell(transform, zs, low_corner, high_corner):
    points = transform(zs)
    return uniform_to_cell(points, low_corner, high_corner)


def uniform_to_cell(zs, low_corner, high_corner, eps=1e-4):
    """Takes points in the uniform distribution and pushes them into the
    cell specified by the given low and high corner."""
    zs = zs * (1 - eps) + 0.5 * eps
    low_corner_array = np.array(low_corner)
    high_corner_array = np.array(high_corner)
    side_lengths = high_corner_array - low_corner_array
    shifted_cell_points = (zs * side_lengths) + low_corner_array
    return shifted_cell_points


def choose_cell(rng, num_cells=100, d=2):
    # For now we assume that num_cells is a perfect square
    num_per_side = round(num_cells ** (1.0 / d))
    side_length = 1.0 / num_per_side
    if not num_per_side ** d == num_cells:
        import pdb

        pdb.set_trace()
    assert num_per_side ** d == num_cells
    cell_idx = random.randint(rng, shape=(1,), minval=0, maxval=num_cells)
    # Need to do this more thoroughly for non two-dimensional
    idxs = np.arange(d)
    temp = cell_idx
    for i in range(d):
        idxs = ops.index_update(idxs, ops.index[i], (temp % num_per_side)[0])
        temp = temp // num_per_side
    low_corner = idxs * side_length
    high_corner = low_corner + side_length
    return low_corner, high_corner


def measure(func):
    """Call by putting the @measure decorator over a function"""

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print("Total execution time: {} ms".format(end_))

    return _time_it


def init_deep_collection(dim, n, num_boxes_per_step, num_cells, rng, betas=None):
    """Makes a set of flow models and returns the list of
    params"""
    lhs = [[]]
    deep_ps_collection = []
    deep_cs_collection = []
    for i in range(num_boxes_per_step):
        l, h = choose_cell(rng, num_cells=num_cells, d=dim)
        rng, _ = random.split(rng, 2)
        lhs.append((l, h))
        if betas is None:
            deep_ps, deep_cs = init_final_sigmoid_nvp_chain(d=dim, n=n, rng=rng)
        else:
            deep_ps, deep_cs = init_final_sigmoid_nvp_chain(
                d=dim, n=n, rng=rng, beta_0s=betas
            )
        deep_ps_collection.append(deep_ps)
        deep_cs_collection.append(deep_cs)
    return deep_ps_collection, deep_cs_collection, lhs[1:]


def init_double_deep_collection(dim, n, num_boxes_per_step, num_cells, rng, betas=None):
    """Makes a set of flow models and returns the list of
    params"""
    lhs = [[]]
    deep_ps_collection = []
    deep_cs_collection = []
    for i in range(num_boxes_per_step):
        l, h = choose_cell(rng, num_cells=num_cells, d=dim)
        rng, _ = random.split(rng, 2)
        lhs.append((l, h))
        if betas is None:
            deep_ps, deep_cs = init_double_sigmoid_nvp_chain(d=dim, n=n, rng=rng)
        else:
            deep_ps, deep_cs = init_double_sigmoid_nvp_chain(
                d=dim, n=n, rng=rng, beta_0s=betas
            )
        deep_ps_collection.append(deep_ps)
        deep_cs_collection.append(deep_cs)
    return deep_ps_collection, deep_cs_collection, lhs[1:]


def init_deep_test_collection(dim, num_boxes_per_step, num_cells, rng):
    """Makes a set of flow models and returns the list of
    params"""
    lhs = [[]]
    deep_ps_collection = []
    deep_cs_collection = []
    for i in range(num_boxes_per_step):
        l, h = choose_cell(rng, num_cells=num_cells, d=dim)
        rng, _ = random.split(rng, 2)
        lhs.append((l, h))
        deep_ps, deep_cs = init_identity_nvp_chain(n=4)
        deep_ps_collection.append(deep_ps)
        deep_cs_collection.append(deep_cs)
    return deep_ps_collection, deep_cs_collection, lhs[1:]


def choose_cells(dim, num_boxes_per_step, num_cells, rng):
    lhs = [[]]
    for i in range(num_boxes_per_step):
        l, h = choose_cell(rng, num_cells=num_cells, d=dim)
        rng, _ = random.split(rng, 2)
        lhs.append((l, h))
    rng, _ = random.split(rng, 2)
    return lhs[1:], rng


def fuse_fn(*args):
    return np.stack(args)


def fuse_params(params_list):
    return tree_multimap(fuse_fn, params_list[0], *params_list[1:])


def unfuse_params(fused_params):
    # Could probably think up a more idiomatic way to do this
    # with treedefs etc
    params_list = [[]]
    for idx in range(len(tree_flatten(fused_params)[0][0])):
        params_list.append(tree_map(lambda x: x[idx], fused_params))
    return params_list[1:]
