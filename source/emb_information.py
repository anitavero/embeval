#!/usr/bin/env python3

""" Demo for (Shannon) mutual information estimators.

Analytical vs estimated value is illustrated for normal random variables.

"""
import os
import numpy as np
from numpy.random import rand, multivariate_normal
from numpy import array, arange, zeros, dot, ones, sum
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import argh
from argh import arg
from typing import List

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_i_shannon

from source.utils import hr_time
from source.process_embeddings import Embeddings


def benchmark(dim=10, cost_name='MIShannon_DKL', num_of_samples=-1, max_num_of_samples=10000):
    """
    Plot estimated vs analytical Mutual Information for random matrices.
    :param dim: Data dimension (number of columns of the matrices)
    :param cost_name: MI estimation algorithm, e.g, 'BIHSIC_IChol', 'MIShannon_DKL', 'MIShannon_HS' (for more see ite.cost)
    :param num_of_samples: if -1 increases the number of data points by 1000 until max_num_of_samples,
                           if >-1 it prints running time for this number of data points (matrix row num)
    :param max_num_of_samples: maximum data point number in case of plotting for a series of sample nums.
    """
    ds = array([dim, dim])  # subspace dimensions: ds[0], ..., ds[M-1]
    if num_of_samples == -1:
        num_of_samples_v = arange(1000, max_num_of_samples + 1, 1000)
    else:
        num_of_samples_v = [num_of_samples]

    # initialization:
    distr = 'normal'  # distribution; fixed
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    # vector of estimated mutual information values:
    i_hat_v = zeros(length)

    # distr, ds -> samples (y), distribution parameters (par), analytical
    # value (i):
    if distr == 'normal':
        dim = sum(ds)  # dimension of the joint distribution

        # mean (m), covariance matrix (c):
        m = rand(dim)
        l = rand(dim, dim)
        c = dot(l, l.T)

        # generate samples (y~N(m,c)):
        y = multivariate_normal(m, c, num_of_samples_max)

        par = {"ds": ds, "cov": c}
    else:
        raise Exception('Distribution=?')

    i = analytical_value_i_shannon(distr, par)

    # estimation:
    total_time = 0
    times = []
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        start_time = time()
        i_hat_v[tk] = co.estimation(y[0:num_of_samples], ds)  # broadcast
        etime = time() - start_time
        print(f"tk={tk + 1}/{length}\tsample num: {num_of_samples}\tTime: {hr_time(etime)}")
        total_time += etime
        times.append(etime)
    print('Total time:', hr_time(total_time))

    if len(num_of_samples_v) > 1:
        # Plot estimation vs analytical
        fig, ax = plt.subplots()
        ax.plot(num_of_samples_v, i_hat_v, num_of_samples_v, ones(length) * i)
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Shannon mutual information')
        ax.legend(('estimation', 'analytical value'), loc='best')
        ax.set_title("Estimator: " + cost_name)
        fig_title = f'Estimator-{cost_name}_dims_{",".join(map(str, ds))}'
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figs', fig_title))

        # Plot time
        fig, ax = plt.subplots()
        ax.plot(num_of_samples_v, times)
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Run time')
        ax.set_title("Time of Estimator: " + cost_name)
        fig_title = f'Time_Estimator-{cost_name}_dims_{",".join(map(str, ds))}'
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figs', fig_title))


def estimate_mi(matrices: List[np.ndarray], cost_name: str):
    """Return estimated Mutual Information for a list of matrices.
        :param matrices: array of numpy matrices
        :param cost_name: MI estimation algorithm, e.g, 'BIHSIC_IChol', 'MIShannon_DKL', 'MIShannon_HS' (for more see ite.cost)
    """
    co = co_factory(cost_name, mult=True)  # cost object
    ds = [m.shape[1] for m in matrices]
    ms_stacked = np.hstack(matrices)
    return co.estimation(ms_stacked, ds)


@arg('-vns', '--vecs_names', nargs='+', type=str, required=True)
def estimate_embeddings_mi(datadir: str, vecs_names=[], cost_name='MIShannon_DKL'):
    """Return estimated Mutual Information for a Embeddings with vecs_names in datadir.
        :param cost_name: MI estimation algorithm, e.g, 'BIHSIC_IChol', 'MIShannon_DKL', 'MIShannon_HS' (for more see ite.cost)
    """
    embs = Embeddings(datadir, vecs_names)
    return estimate_mi(embs.embeddings, cost_name)


if __name__ == "__main__":
    argh.dispatch_commands([benchmark, estimate_embeddings_mi])
