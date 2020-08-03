#!/usr/bin/env python3

""" Demo for (Shannon) mutual information estimators.

Analytical vs estimated value is illustrated for normal random variables.

"""
import os
from numpy.random import rand, multivariate_normal
from numpy import array, arange, zeros, dot, ones, sum
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import argh
from argh import arg
from glob import glob
import json

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_i_shannon

from source.utils import hr_time, tuple_list
from source.process_embeddings import Embeddings, mid_fusion


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


@arg('-mmembs', '--mm_embs_of', type=tuple_list)
@arg('-vns', '--vecs_names', nargs='+', type=str, required=True)
def estimate_embeddings_mi(datadir: str, vecs_names=[], mm_embs_of=[], cost_name='MIShannon_DKL'):
    """Return estimated Mutual Information for a Embeddings with vecs_names in datadir.
        :param datadir: Path to directory which contains embedding data.
        :param vecs_names: List[str] Names of embeddings
        :param mm_embs_of: List of str tuples, where the tuples contain names of embeddings which are to
                       be concatenated into a multi-modal mid-fusion embedding.
        :param cost_name: MI estimation algorithm, e.g, 'BIHSIC_IChol', 'MIShannon_DKL', 'MIShannon_HS' (for more see ite.cost)
    """
    embs = Embeddings(datadir, vecs_names)
    emb_tuples = [tuple(embs.embeddings[vecs_names.index(l)] for l in t) for t in mm_embs_of]
    vocab_tuples = [tuple(embs.vocabs[vecs_names.index(l)] for l in t) for t in mm_embs_of]
    mm_labels = [tuple(l for l in t) for t in mm_embs_of]
    mm_embeddings, mm_vocabs, mm_labels = mid_fusion(emb_tuples, vocab_tuples, mm_labels, padding=False)

    # Compute estimates MI for all multi-modal embeddings
    print('Compute Mutual Information...')
    eMIs = {}
    for mme, mml in zip(mm_embeddings, mm_embs_of):
        print(mml)
        co = co_factory(cost_name, mult=True)  # cost object
        ds = [embs.embeddings[vecs_names.index(l)].shape[1] for l in mml]
        eMIs['-'.join(mml)] = co.estimation(mme, ds)
        print(eMIs['-'.join(mml)])

    return eMIs


@arg('-exp', '--exp_names', nargs='+', type=str, required=True)
def run_mi_experiments(exp_names='quantity'):
    embdir = '/anfs/bigdisc/alv34/wikidump/extracted/models/'
    savedir = embdir + '/results/'

    if 'quantity' in exp_names:
        models = glob(embdir + '*model*npy*')
        ling_names = [os.path.split(m)[1].split('.')[0] for m in models if 'fqrng' not in m]
        vis_names = ['vecs3lem1', 'google_resnet152']
        mm_embs = [(l, v) for l in ling_names for v in vis_names]
        MIs = estimate_embeddings_mi(embdir, vecs_names=ling_names + vis_names,
                                     mm_embs_of=mm_embs, cost_name='MIShannon_DKL')

        with open(os.path.join(savedir, 'MM_MI_for_quantities.json'), 'w') as f:
            json.dump(MIs, f)


if __name__ == "__main__":
    argh.dispatch_commands([benchmark, estimate_embeddings_mi, run_mi_experiments])
