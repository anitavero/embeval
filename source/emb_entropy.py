#!/usr/bin/env python3

""" Demo for KL divergence estimators.

Aanalytical vs estimated value is illustrated for normal random variables.

"""
from numpy.random import rand, multivariate_normal
from numpy import arange, zeros, dot, ones
import matplotlib.pyplot as plt
from tqdm import tqdm
import argh
from argh import arg

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_d_kullback_leibler


def run_benchmark(dim, k):
    """"
    :param dim: dimension of the distribution
    :param k: number of nearest neighbours
    """
    num_of_samples_v = arange(1000, 10 * 1000 + 1, 1000)
    cost_name = 'BDKL_KnnK'  # dim >= 1
    # cost_name = 'BDKL_KnnKiTi'  # dim >= 1
    # cost_name = 'MDKL_HSCE'  # dim >= 1

    # initialization:
    distr = 'normal'  # fixed
    num_of_samples_max = num_of_samples_v[-1]
    co = co_factory(cost_name, mult=True, k=k)  # cost object

    # distr, dim -> samples (y1,y2), distribution parameters (par1,par2),
    # analytical value (d):
    if distr == 'normal':
        # mean (m1,m2):
        m2 = rand(dim)
        m1 = m2

        # (random) linear transformation applied to the data (l1,l2) ->
        # covariance matrix (c1,c2):
        l2 = rand(dim, dim)
        l1 = rand(1) * l2
        # Note: (m2,l2) => (m1,l1) choice guarantees y1<<y2
        # (in practise, too).

        c1 = dot(l1, l1.T)
        c2 = dot(l2, l2.T)

        # generate samples (y1~N(m1,c1), y2~N(m2,c2)):
        y1 = multivariate_normal(m1, c1, num_of_samples_max)
        y2 = multivariate_normal(m2, c2, num_of_samples_max)

        par1 = {"mean": m1, "cov": c1}
        par2 = {"mean": m2, "cov": c2}
    else:
        raise Exception('Distribution=?')

    d = analytical_value_d_kullback_leibler(distr, distr, par1, par2)

    # estimation:
    d_hat_v = co.estimation(y1, y2)
    relative_err = abs(d_hat_v - d) / d
    return relative_err


@arg('dim', type=int)
@arg('sample_num', type=int)
def benchmark(dim, sample_num):
    mean_rel_errs = {}
    for k in tqdm([3, 5, 10]):
        sum_rel_errs = 0
        for i in tqdm(range(sample_num)):
            sum_rel_errs += run_benchmark(dim, k)
        mean_rel_errs[k] = sum_rel_errs / sample_num
    for k, rerr in mean_rel_errs.items():
        print('k:', k, ', Mean Relative Error:', rerr)


if __name__ == "__main__":
    argh.dispatch_command(benchmark)
