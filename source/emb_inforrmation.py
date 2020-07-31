#!/usr/bin/env python3

""" Demo for (Shannon) mutual information estimators.

Analytical vs estimated value is illustrated for normal random variables.

"""

from numpy.random import rand, multivariate_normal
from numpy import array, arange, zeros, dot, ones, sum
import matplotlib.pyplot as plt
from time import time
import argh

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_i_shannon

from source.utils import hr_time


def main(dim=10):
    # parameters:
    ds = array([dim, dim])  # subspace dimensions: ds[0], ..., ds[M-1]
    num_of_samples_v = arange(1000, 10 * 1000 + 1, 1000)

    cost_name = 'BIHSIC_IChol'  # d_m >= 1, M >= 2
    # cost_name = 'MIShannon_DKL'  # d_m >= 1, M >= 2
    # cost_name = 'MIShannon_HS'  # d_m >= 1, M >= 2

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

    # Plot estimation vs analytical
    fig, ax = plt.subplots()
    ax.plot(num_of_samples_v, i_hat_v, num_of_samples_v, ones(length) * i)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Shannon mutual information')
    ax.legend(('estimation', 'analytical value'), loc='best')
    ax.set_title("Estimator: " + cost_name)
    fig_title = f'Estimator-{cost_name}_dims_{",".join(map(str, ds))}'
    plt.savefig('../figs/' + fig_title)

    # Plot time
    fig, ax = plt.subplots()
    ax.plot(num_of_samples_v, times)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Run time')
    ax.set_title("Time of Estimator: " + cost_name)
    fig_title = f'Time_Estimator-{cost_name}_dims_{",".join(map(str, ds))}'
    plt.savefig('../figs/' + fig_title)


if __name__ == "__main__":
    argh.dispatch_command(main)
