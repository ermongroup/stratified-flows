import jax.numpy as np
from jax import vmap
import jax.random as random
import numpy as onp
from jax.scipy.special import logsumexp
from uniform_flow_model import sample_our_uniform
import matplotlib.pyplot as plt


def new_log_integrand_itemwise(y, x, params, sigma):
    """Params is a vector of size (k), even integer k"""
    k = len(params)
    a_array = params[:k // 2]
    b_array = params[k // 2:]
    log_regularization_term = -((params * params) / (2 * 9)).sum()
    quadratic_terms = -0.5 * ((y - (a_array * x + b_array))**2 / (sigma**2))
    log_prefactor = -np.log(2 * np.pi) - np.log(sigma) - np.log(k // 2)  # prior on h
    return log_prefactor + logsumexp(quadratic_terms) + log_regularization_term


vmapped_log_integrand_itemwise = vmap(new_log_integrand_itemwise, in_axes=(0, 0,
                                                                           None, None))


def log_integrand(y, x, params, sigma):
    log_losses = vmapped_log_integrand_itemwise(y, x, params, sigma)
    return np.sum(log_losses)


def generate_data(params, sigma):
    """Generates the data for the Bayesian problem"""
    rng = random.PRNGKey(0)
    k = len(params) // 2
    a_array = params[:k]
    b_array = params[k:]
    n = 20 * k
    xs = sample_our_uniform(n, 1, rng).reshape((n,))
    ys = onp.zeros(n)
    all_indices = set(onp.arange(n))
    for i in range(k):
        i_idxs = onp.random.choice(list(all_indices), 20, replace=False)
        all_indices = set(all_indices) - set(i_idxs)
        ys[i_idxs] = xs[i_idxs] * a_array[i] + b_array[i] + onp.random.normal(0, sigma, size=(20,))
    return xs, ys


def plot_points():
    onp.random.seed(0)
    params = np.array([0, 1, -1, 0.5,
                       2, 0, -1, -2])
    x, y = generate_data(params, 0.1)
    np.save('x.npy', x)
    np.save('y.npy', y)
    plt.plot(x, y, 'k+', markersize=5)
    plt.show()


def plot_loaded_points():
    onp.random.seed(0)
    params = np.array([0, 1, -1, 0.5,
                       2, 0, -1, -2])
    x, y = generate_data(params, 0.1)
    x = np.load('x.npy')
    y = np.load('y.npy')
    fig, ax = plt.subplots()
    for i in range(4): ax.plot([-1, 2], [-1 * params[i] + params[i + 4],
                                         2 * params[i] + params[i + 4]], 'k-')
    ax.set_xlim([0, 1])
    ax.set_ylim([-3, 3])
    ax.plot(x, y, 'k+', markersize=10)
    ax.set_ylabel(r'y')
    ax.set_xlabel(r'x')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
