import jax.numpy as np
import jax.random as random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
from collections import namedtuple


# This serves as the core of the setup.
# Each layer is represented with a (Parameters, Layer)
# pair, where Layer contains the fn if it's a scale
# and shift, whether to flip, and whether to use the
# inverse sigmoid, regular sigmoid, or scale-and-shift.
Layer = namedtuple("Layer", "fn flip sigmoid inv_sigmoid")


def sample_our_uniform(N, d, rng, tol=1e-4):
    """Sample from the uniform distribution, cutting off at
    0.5 * tol"""
    return random.uniform(rng, (N, d), minval=0, maxval=1) * (1 - tol) + 0.5 * tol


def fused_sample_our_uniform(N, num_batches, d, rng, tol):
    """Samples batches from the uniform distribution,
    with output shape [num_batches x N x d]"""
    flat_batches = sample_our_uniform(N * num_batches, d, rng, tol)
    return flat_batches.reshape((num_batches, N, d))


def constant_shift_log_scale_fn(params, y1):
    """For testing"""
    dim = y1.shape[-1]
    return np.ones(dim) * -2, np.ones(dim) * np.log(2)


def identity_scale_fn(params, y1):
    return np.zeros_like(y1), np.zeros_like(y1)


def zero_base_fn(x):
    return np.ones(len(x)) * 0


def log_prob_our_uniform(x):
    """With  exponential (log-linear) tails"""
    # x = x + np.ones(2) * -0.5
    linear_slope = 20
    const_log_density = 0
    x_out_of_bounds = np.any(np.abs(x - 0.5) > 0.5, axis=1)
    x_in_bounds = x_out_of_bounds * (1 - x_out_of_bounds) + (
        1 - x_out_of_bounds
    )  # swap sign
    in_bounds_probs = const_log_density * np.ones((np.shape(x)[0],)) * x_in_bounds
    out_of_bounds_probs_gr = np.sum((x > 1) * (1 - x) * linear_slope, axis=1)
    out_of_bounds_probs_le = np.sum((x < 0) * x * linear_slope, axis=1)
    out_of_bounds_probs = out_of_bounds_probs_gr + out_of_bounds_probs_le
    return in_bounds_probs + out_of_bounds_probs


def nvp_forward(net_params, shift_and_log_scale_fn, x, flip=False):
    d = x.shape[-1] // 2
    x1, x2 = x[:, :d], x[:, d:]
    if flip:
        x2, x1 = x1, x2
    shift, log_scale = shift_and_log_scale_fn(net_params, x1)
    y2 = x2 * np.exp(log_scale) + shift
    if flip:
        x1, y2 = y2, x1
    y = np.concatenate([x1, y2], axis=-1)
    return y, log_scale


def nvp_inverse(net_params, shift_and_log_scale_fn, y, flip=False):
    d = y.shape[-1] // 2
    y1, y2 = y[:, :d], y[:, d:]
    if flip:
        y1, y2 = y2, y1
    shift, log_scale = shift_and_log_scale_fn(net_params, y1)
    x2 = (y2 - shift) * np.exp(-log_scale)
    if flip:
        y1, x2 = x2, y1
    x = np.concatenate([y1, x2], axis=-1)
    return x, log_scale


def sigmoid_inverse(params, y):
    """Maps y in (0, 1)^d to (-infinity, infinity)^d"""
    beta, log_scale, shift = params
    inv_sigmoid_log_prob = np.sum(
        -np.log(beta) - np.log(y) - np.log(1 - y) - log_scale, axis=1
    )
    y = (1.0 / beta) * (np.log(y) - np.log(1 - y))
    y = y * np.exp(-log_scale) - shift
    return y, inv_sigmoid_log_prob


def sigmoid_forward(params, x):
    """Maps x in (-infinity, infinity)^d to (0, 1)^d"""
    beta, log_scale, shift = params
    x = (x + shift) * np.exp(log_scale)  # To get closer to unit square when mapped out
    sigmoid_log_prob = np.sum(
        log_scale + np.log(beta) - beta * x - 2 * np.logaddexp(0, -beta * x), axis=1
    )
    x = 1.0 / (1 + np.exp(-beta * x))
    return x, sigmoid_log_prob


def sample_nvp(net_params, shift_log_scale_fn, base_sample_fn, N, flip=False):
    x = base_sample_fn(N)
    return nvp_forward(net_params, shift_log_scale_fn, x, flip=flip)


def log_prob_nvp(
    net_params,
    shift_log_scale_fn,
    log_prob_our_uniform,
    y,
    flip=False,
    sigmoid=False,
    inv_sigmoid=False,
):
    """This calculates the reverse log prob, so if we have a sigmoid
    we have to compute the inverse sigmoid as we're going backwards"""
    if sigmoid:
        x, ildj = sigmoid_inverse(net_params, y)
    elif inv_sigmoid:
        x, ildj = sigmoid_forward(net_params, y)
    else:
        x, log_scale = nvp_inverse(net_params, shift_log_scale_fn, y, flip=flip)
        ildj = -np.sum(log_scale, axis=-1)
    return log_prob_our_uniform(x) + ildj


def forward_log_prob_nvp(
    net_params,
    shift_log_scale_fn,
    log_prob_fn,
    x_1,
    flip=False,
    sigmoid=False,
    inv_sigmoid=False,
):
    if sigmoid:
        x_2, ldj = sigmoid_forward(net_params, x_1)
    elif inv_sigmoid:
        x_2, ldj = sigmoid_inverse(net_params, x_1)
    else:
        x_2, log_scale = nvp_forward(net_params, shift_log_scale_fn, x_1, flip=flip)
        ldj = np.sum(log_scale, axis=-1)
    return log_prob_fn(x_2) + ldj


def init_nvp(D_in, D_out, rng):
    net_init, net_apply = stax.serial(
        Dense(256), Relu, Dense(256), Relu, Dense(D_out * 2)
    )  # 2 for scale & shift
    in_shape = (-1, D_in)
    out_shape, net_params = net_init(rng, in_shape)

    def shift_and_log_scale_fn(net_params, x1):
        s = net_apply(net_params, x1)
        return np.split(s, 2, axis=1)

    return net_params, shift_and_log_scale_fn


def init_nvp_chain(d, n, rng):
    k = d // 2
    if d % 2 == 0:
        d_1_in, d_1_out = k, k
        d_2_in, d_2_out = k, k
    else:
        # d = 2k + 1
        d_1_in, d_1_out = k, k + 1
        d_2_in, d_2_out = k + 1, k
    flip = False
    ps, configs = [], []
    for i in range(n):
        if flip:
            p, f = init_nvp(d_2_in, d_2_out, rng)
        else:
            p, f = init_nvp(d_1_in, d_1_out, rng)
        rng, _ = random.split(rng, 2)
        layer = Layer(fn=f, flip=flip, sigmoid=False, inv_sigmoid=False)
        ps.append(p), configs.append(layer)
        flip = not flip
    return ps, configs


def init_final_sigmoid_nvp_chain(d, n, rng, beta_0s=np.array([1.0, 1.6, -0.5])):
    beta, log_scale, shift = beta_0s
    k = d // 2
    if d % 2 == 0:
        d_1_in, d_1_out = k, k
        d_2_in, d_2_out = k, k
    else:
        # d = 2k + 1
        d_1_in, d_1_out = k, k + 1
        d_2_in, d_2_out = k + 1, k
    flip = False
    ps, configs = [], []
    for i in range(n):
        if flip:
            p, f = init_nvp(d_2_in, d_2_out, rng)
        else:
            p, f = init_nvp(d_1_in, d_1_out, rng)
        rng, _ = random.split(rng, 2)
        layer = Layer(fn=f, flip=flip, sigmoid=False, inv_sigmoid=False)
        ps.append(p), configs.append(layer)
        flip = not flip
    final_layer = Layer(fn=None, flip=flip, sigmoid=True, inv_sigmoid=False)
    ps.append(np.array([beta, log_scale, shift])), configs.append(final_layer)
    return ps, configs


def init_initial_inv_sigmoid_nvp_chain(d, n, rng, beta_0s=np.array([1.0, 1.6, -0.5])):
    """This chain initially uses an inverse sigmoid, so goes from
    (0, 1)^d to (-infinity, infinity)^d"""
    beta, log_scale, shift = beta_0s
    k = d // 2
    if d % 2 == 0:
        d_1_in, d_1_out = k, k
        d_2_in, d_2_out = k, k
    else:
        # d = 2k + 1
        d_1_in, d_1_out = k, k + 1
        d_2_in, d_2_out = k + 1, k
    flip = False
    ps, configs = [[]], []
    first_layer = Layer(fn=None, flip=flip, sigmoid=False, inv_sigmoid=True)
    ps.append(np.array([beta, log_scale, shift])), configs.append(first_layer)
    for i in range(n):
        if flip:
            p, f = init_nvp(d_2_in, d_2_out, rng)
        else:
            p, f = init_nvp(d_1_in, d_1_out, rng)
        rng, _ = random.split(rng, 2)
        layer = Layer(fn=f, flip=flip, sigmoid=False, inv_sigmoid=False)
        ps.append(p), configs.append(layer)
        flip = not flip
    return ps[1:], configs


def init_double_sigmoid_nvp_chain(d, n, rng, beta_0s=np.array([1.0, 1.6, -0.5])):
    """This chain initially uses an inverse sigmoid, so goes from
    (0, 1)^d to (-infinity, infinity)^d, then has a few layers, then
    goes back through a sigmoid"""
    beta, log_scale, shift = beta_0s
    k = d // 2
    if d % 2 == 0:
        d_1_in, d_1_out = k, k
        d_2_in, d_2_out = k, k
    else:
        # d = 2k + 1
        d_1_in, d_1_out = k, k + 1
        d_2_in, d_2_out = k + 1, k
    flip = False
    ps, configs = [[]], []
    first_layer = Layer(fn=None, flip=flip, sigmoid=False, inv_sigmoid=True)
    ps.append(np.array([beta, log_scale, shift])), configs.append(first_layer)
    for i in range(n):
        if flip:
            p, f = init_nvp(d_2_in, d_2_out, rng)
        else:
            p, f = init_nvp(d_1_in, d_1_out, rng)
        rng, _ = random.split(rng, 2)
        layer = Layer(fn=f, flip=flip, sigmoid=False, inv_sigmoid=False)
        ps.append(p), configs.append(layer)
        flip = not flip
    final_layer = Layer(fn=None, flip=flip, sigmoid=True, inv_sigmoid=False)
    ps.append(np.array([beta, log_scale, shift])), configs.append(final_layer)
    return ps[1:], configs


def init_identity_double_sigmoid_nvp_chain(
    d, n, rng, beta_0s=np.array([1.0, 1.6, -0.5])
):
    """This chain initially uses an inverse sigmoid, so goes from
    (0, 1)^d to (-infinity, infinity)^d, then has a few layers, then
    goes back through a sigmoid"""
    beta, log_scale, shift = beta_0s
    flip = False
    ps, configs = [[]], []
    first_layer = Layer(fn=None, flip=flip, sigmoid=False, inv_sigmoid=True)
    ps.append(np.array([beta, log_scale, shift])), configs.append(first_layer)
    for i in range(n):
        f = identity_scale_fn
        layer = Layer(fn=f, flip=flip, sigmoid=False, inv_sigmoid=False)
        ps.append([1.0, 1.0]), configs.append(layer)
        flip = not flip
    final_layer = Layer(fn=None, flip=flip, sigmoid=True, inv_sigmoid=False)
    ps.append(np.array([beta, log_scale, shift])), configs.append(final_layer)
    return ps[1:], configs


def init_constant_nvp_chain(n=2):
    flip = False
    ps, configs = [], []
    for i in range(n):
        f = constant_shift_log_scale_fn
        layer = Layer(fn=f, flip=flip, sigmoid=False, inv_sigmoid=False)
        ps.append(["none"]), configs.append(layer)
        flip = not flip
    return ps, configs


def init_identity_nvp_chain(n=2):
    flip = False
    ps, configs = [[]], []
    for i in range(n):
        f = identity_scale_fn
        layer = Layer(fn=f, flip=flip, sigmoid=False, inv_sigmoid=False)
        ps.append([1.0, 1.0]), configs.append(layer)
        flip = not flip
    return ps[1:], configs


def sample_nvp_chain(ps, configs, base_sample_fn, N, rng):
    x = base_sample_fn(N, rng)
    for p, config in zip(ps, configs):
        shift_log_scale_fn = config.fn
        flip = config.flip
        sigmoid = config.sigmoid
        inv_sigmoid = config.inv_sigmoid
        x = nvp_forward(
            p,
            shift_log_scale_fn,
            x,
            flip=flip,
            sigmoid=sigmoid,
            inv_sigmoid=inv_sigmoid,
        )
    return x


def make_log_prob_fn(p, log_prob_fn, config):
    shift_log_scale_fn = config.fn
    flip = config.flip
    sigmoid = config.sigmoid
    inv_sigmoid = config.inv_sigmoid
    return lambda x: log_prob_nvp(
        p,
        shift_log_scale_fn,
        log_prob_fn,
        x,
        flip=flip,
        sigmoid=sigmoid,
        inv_sigmoid=inv_sigmoid,
    )


def make_forward_log_prob_fn(p, log_prob_fn, config):
    shift_log_scale_fn = config.fn
    flip = config.flip
    sigmoid = config.sigmoid
    inv_sigmoid = config.inv_sigmoid

    return lambda x: forward_log_prob_nvp(
        p,
        shift_log_scale_fn,
        log_prob_fn,
        x,
        flip=flip,
        sigmoid=sigmoid,
        inv_sigmoid=inv_sigmoid,
    )


def log_prob_nvp_chain(ps, configs, log_prob_our_uniform, y):
    log_prob_fn = log_prob_our_uniform
    for p, config in zip(ps, configs):
        log_prob_fn = make_log_prob_fn(p, log_prob_fn, config)
    return log_prob_fn(y)


def forward_log_prob_nvp_chain(ps, configs, log_prob_fn_1, x):
    log_prob_fn = log_prob_fn_1  # Avoid the 'arguments-instantiated' bug
    for p, config in zip(ps[::-1], configs[::-1]):
        log_prob_fn = make_forward_log_prob_fn(p, log_prob_fn, config)
    return log_prob_fn(x)


def log_det_J_u(ps, configs, x):
    return forward_log_prob_nvp_chain(ps, configs, zero_base_fn, x)


def pushforward(ps, cs, w):
    for p, config in zip(ps, cs):
        shift_log_scale_fn = config.fn
        flip = config.flip
        sigmoid = config.sigmoid
        inv_sigmoid = config.inv_sigmoid

        if sigmoid:
            w, _ = sigmoid_forward(p, w)
        elif inv_sigmoid:
            w, _ = sigmoid_inverse(p, w)
        else:
            w, _ = nvp_forward(p, shift_log_scale_fn, w, flip=flip)
    return w


def pullback(ps, cs, x):
    for p, config in zip(ps[::-1], cs[::-1]):
        shift_log_scale_fn = config.fn
        flip = config.flip
        sigmoid = config.sigmoid
        inv_sigmoid = config.inv_sigmoid

        if sigmoid:
            x, _ = sigmoid_inverse(p, x)
        elif inv_sigmoid:
            x, _ = sigmoid_forward(p, x)
        else:
            x, _ = nvp_inverse(p, shift_log_scale_fn, x, flip=flip)
    return x


def staggered_pushforward(ps, cs, x):
    values = []
    for p, config in zip(ps, cs):
        shift_log_scale_fn = config.fn
        flip = config.flip
        sigmoid = config.sigmoid
        inv_sigmoid = config.inv_sigmoid
        if sigmoid:
            x, _ = sigmoid_forward(p, x)
        elif inv_sigmoid:
            x, _ = sigmoid_inverse(p, x)
        else:
            x, _ = nvp_forward(p, shift_log_scale_fn, x, flip=flip)
        values.append(x)
    return values
