#!/usr/bin/env python3

"""
Module for (Shannon) mutual information estimators.
"""

import os
from numpy.random import rand, multivariate_normal
from numpy import array, arange, zeros, dot, ones, sum
import matplotlib
matplotlib.rcParams["savefig.dpi"] = 300
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')
from time import time
import argh
from argh import arg
from tqdm import tqdm
from glob import glob
import json
import numpy as np
from sklearn.decomposition import PCA

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_i_shannon
from ite.cost.x_kernel import Kernel

from source.utils import hr_time, tuple_list
from source.process_embeddings import Embeddings, mid_fusion, MM_TOKEN


FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figs')


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
        plt.savefig(os.path.join(FIG_DIR, fig_title))

        # Plot time
        fig, ax = plt.subplots()
        ax.plot(num_of_samples_v, times)
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Run time')
        ax.set_title("Time of Estimator: " + cost_name)
        fig_title = f'Time_Estimator-{cost_name}_dims_{",".join(map(str, ds))}'
        plt.savefig(os.path.join(FIG_DIR, fig_title))


@arg('-mmembs', '--mm_embs_of', type=tuple_list)
@arg('-n-pca', '--pca_n_components', type=int)
@arg('-vns', '--vecs_names', nargs='+', type=str, required=True)
def estimate_embeddings_mi(datadir: str, vecs_names=[], mm_embs_of=[], cost_name='MIShannon_DKL',
                           pca_n_components=None):
    """
Return estimated Mutual Information for a Embeddings with vecs_names in datadir.
        :param datadir: Path to directory which contains embedding data.
        :param vecs_names: List[str] Names of embeddings
        :param mm_embs_of: List of str tuples, where the tuples contain names of embeddings which are to
                       be concatenated into a multi-modal mid-fusion embedding.
        :param cost_name: MI estimation algorithm, e.g, 'BIHSIC_IChol', 'MIShannon_DKL', 'MIShannon_HS' (for more see ite.cost)
    """
    embs = Embeddings(datadir, vecs_names)

    var_ratios = {}
    if pca_n_components:
        small_embs = []
        print(f'Reduce dimension to {pca_n_components} with PCA...')
        for e, nm in tqdm(zip(embs.embeddings, embs.vecs_names)):
            e_small, var_ratio = run_pca(e, pca_n_components)
            small_embs.append(e_small)
            var_ratios[nm] = list(var_ratio)
        embs.embeddings = small_embs

    emb_tuples = [tuple(embs.embeddings[vecs_names.index(l)] for l in t) for t in mm_embs_of]
    vocab_tuples = [tuple(embs.vocabs[vecs_names.index(l)] for l in t) for t in mm_embs_of]
    mm_labels = [tuple(l for l in t) for t in mm_embs_of]
    mm_embeddings, mm_vocabs, mm_labels = mid_fusion(emb_tuples, vocab_tuples, mm_labels, padding=False)

    # Compute estimates MI for all multi-modal embeddings
    print('Compute Mutual Information...')
    eMIs = {}
    for mme, mml in zip(mm_embeddings, mm_embs_of):
        print(mml)
        k = Kernel({'name': 'RBF', 'sigma': 'median'})
        co = co_factory(cost_name, mult=True, kernel=k)  # cost object
        ds = [embs.embeddings[vecs_names.index(l)].shape[1] for l in mml]
        eMIs[MM_TOKEN.join(mml)] = co.estimation(mme, ds)
        print(eMIs[MM_TOKEN.join(mml)])

    return eMIs, var_ratios


def run_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_small = pca.fit_transform(X)
    return X_small, pca.explained_variance_ratio_


@arg('-exp', '--exp_names', nargs='+', type=str, required=True)
@arg('-n-pca', '--pca_n_components', type=int)
def run_mi_experiments(exp_names='quantity', cost_name='MIShannon_DKL', pca_n_components=None, exp_suffix='',
                       blas_n_threads=None):
    """
    :param cost_name: MI estimation algorithm, e.g, HSIC kernel method: 'BIHSIC_IChol',
                                                    KNN based linear:   'MIShannon_DKL'
    """
    if blas_n_threads:
        os.environ["OPENBLAS_NUM_THREADS"] = blas_n_threads

    embdir = '/anfs/bigdisc/alv34/wikidump/extracted/models/'
    savedir = embdir + '/results/'

    vis_names = ['vecs3lem1', 'google_resnet152']
    models = glob(embdir + '*model*npy*')

    if exp_suffix != '' and exp_suffix[0] != '_':
        exp_suffix = '_' + exp_suffix

    if 'quantity' in exp_names:
        ling_names = [os.path.split(m)[1].split('.')[0] for m in models if 'fqrng' not in m]
        mm_embs = [(l, v) for l in ling_names for v in vis_names]
        MIs, var_ratios = estimate_embeddings_mi(embdir, vecs_names=ling_names + vis_names,
                                     mm_embs_of=mm_embs, cost_name=cost_name, pca_n_components=pca_n_components)

        with open(os.path.join(savedir, f'MM_MI_{cost_name}_for_quantities{exp_suffix}.json'), 'w') as f:
            json.dump(MIs, f)
        if var_ratios != {}:
            with open(os.path.join(savedir, f'MM_MI_{cost_name}_for_quantities{exp_suffix}_var-ratios.json'), 'w') as f:
                json.dump(var_ratios, f)

    if 'freqranges' in exp_names:
        ling_names = [os.path.split(m)[1].split('.')[0] for m in models if 'fqrng' in m or 'n-1' in m]
        mm_embs = [(l, v) for l in ling_names for v in vis_names]
        MIs, var_ratios = estimate_embeddings_mi(embdir, vecs_names=ling_names + vis_names,
                                     mm_embs_of=mm_embs, cost_name=cost_name, pca_n_components=pca_n_components)

        with open(os.path.join(savedir, f'MM_MI_{cost_name}_for_freqranges{exp_suffix}.json'), 'w') as f:
            json.dump(MIs, f)
        if var_ratios != {}:
            with open(os.path.join(savedir, f'MM_MI_{cost_name}_for_freqranges{exp_suffix}_var-ratios.json'), 'w') as f:
                json.dump(var_ratios, f)


LABELS = {'vecs3lem1': r'I($E_L,E_S$)', 'google_resnet152': r'I($E_L,E_V$)'}
bar_width = 0.4
fontsize = 20

def plot_for_quantities(file_path, vis_names=['vecs3lem1', 'google_resnet152'], legend=True, fname='', suffix=''):
    with open(file_path, 'r') as f:
        MIs = json.load(f)
    quantities = sorted(list(set([int(n.split('_')[1][1:]) for n in MIs.keys()])))
    quantities = quantities[1:] + [quantities[0]]  # -1: max train file num

    # Plot data with error bars
    def bar_data(nms):
        means, errs = [], []
        for q in quantities:
            qnames = [n for n in nms if f'n{q}_' in n]
            qMIs = [v for k, v in MIs.items() if k in qnames]
            q_mean, q_std = np.mean(qMIs), np.std(qMIs)
            means.append(q_mean)
            errs.append(q_std)
        return means, errs

    fig, ax = plt.subplots()
    xpos = np.linspace(1, 2 + 2 * len(vis_names), len(quantities))

    for i, vn in enumerate(vis_names):
        vnms = [k for k in MIs.keys() if vn in k]
        means, errs = bar_data(vnms)
        ax.bar(np.array(xpos) + i * bar_width, means, yerr=errs, width=bar_width, label=LABELS[vn])

    ax.set_xticks(xpos)
    ax.set_xticklabels(['8M', '1G', '2G', '5G', '13G'], fontsize=fontsize)
    ax.set_ylabel('Mutual Information', fontsize=fontsize)
    if legend:
        ax.legend(loc='best', fontsize=fontsize)
    if suffix != '' and suffix[0] != '_': suffix = '_' + suffix
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.savefig(os.path.join(FIG_DIR, f'{fname}{suffix}'), bbox_inches='tight')


def plot_for_freqranges(file_path, vis_names=['vecs3lem1', 'google_resnet152'], quantity=-1, legend=True, fname='',
                        suffix=''):
    with open(file_path, 'r') as f:
        MIs = json.load(f)
    freq_ranges = sorted(list(set([tuple(map(int, n.split('_')[-1].split(MM_TOKEN)[0].split('-'))) for n in MIs.keys()
                                   if f'n{quantity}' in n and 'fqrng' in n and 'vecs3lem1' in n])))
    mixed_names = [n for n in MIs.keys() if 'fqrng' not in n and 'n-1' in n]

    # Plot data with error bars
    def bar_data(nms):
        means, errs = [], []
        for fmin, fmax in freq_ranges:
            fnames = [n for n in nms if f'fqrng_{fmin}-{fmax}' in n]
            fMIs = [v for k, v in MIs.items() if k in fnames]
            f_mean, f_std = np.mean(fMIs), np.std(fMIs)
            means.append(f_mean)
            errs.append(f_std)
        mMIs = [v for k, v in MIs.items() if k in mixed_names and k in nms]
        m_mean, m_std = np.mean(mMIs), np.std(mMIs)
        means.append(m_mean)
        errs.append(m_std)
        return means, errs

    fig, ax = plt.subplots()
    xpos = np.linspace(1, 2 + 2 * len(vis_names), len(freq_ranges) + 1)

    for i, vn in enumerate(vis_names):
        vnms = [k for k in MIs.keys() if vn in k and f'n{quantity}' in k]
        means, errs = bar_data(vnms)
        ax.bar(np.array(xpos) + i * bar_width, means, yerr=errs, width=bar_width, label=LABELS[vn])

    ax.set_xticks(xpos)
    ax.set_xticklabels(['LOW', 'MEDIUM', 'HIGH', 'MIXED'], fontsize=fontsize)
    ax.set_ylabel('Mutual Information', fontsize=fontsize)
    if legend:
        ax.legend(loc='best', fontsize=fontsize)
    if suffix != '' and suffix[0] != '_': suffix = '_' + suffix
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.savefig(os.path.join(FIG_DIR, f'{fname}{suffix}'), bbox_inches='tight')


def plots(file_pattern, vis_names=['vecs3lem1', 'google_resnet152'], fqrng_quantity=-1, legend=True, suffix=''):
    files = glob(file_pattern)
    qfiles = [f for f in files if 'quantities' in f]
    ffiles = [f for f in files if 'freqranges' in f]

    for f in tqdm(qfiles, desc='Quantities'):
        print(f)
        fname = os.path.basename(f).split('.')[0]
        plot_for_quantities(f, vis_names=vis_names, legend=legend, fname=fname, suffix=suffix)
    for f in tqdm(ffiles, desc='Freq ranges'):
        print(f)
        fname = os.path.basename(f).split('.')[0]
        plot_for_freqranges(f, vis_names=vis_names, legend=legend, fname=fname, suffix=suffix, quantity=fqrng_quantity)


if __name__ == "__main__":
    argh.dispatch_commands([benchmark, estimate_embeddings_mi, run_mi_experiments, plot_for_quantities,
                            plot_for_freqranges, plots])
