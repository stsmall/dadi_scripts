#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:09:14 2017

@author: scott
"""
import numpy as np
import dadi
import demographic_models
import pylab
import argparse
from collections import defaultdict
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--infile", type=str, required=True,
                    help='infile name')
parser.add_argument('-p', "--pop", nargs='+', required=True,
                    help='name pop1 pop2')
parser.add_argument('-s', "--size", nargs='+', required=True,
                    help='size pop1 pop2')
parser.add_argument("--totalsize", default=60E6, required=False,
                    help='total genome size')
parser.add_argument("--chunksize", default=2E6, required=False,
                    help='chunked size')
parser.add_argument("--boots", default=100, required=False,
                    help='number bootstraps')
parser.add_argument("--lrt", action="store_true",
                    help="run LRT between multiple models")
args = parser.parse_args()


def load_data(infile, pops, sizes, fold=True, mask=True):
    """Import the standard dadi input file. This file can be constructed with
    the script vcf2dadi in the same repository

    Parameters
    ------
    infile: file, dadi formatted file
    fold: bool, True is folded, default

    Returns
    ------
    fs: freq spectrum object
    dd: data_dict

    """
    print("loading data ...")
    # sample sizes in diploid
    # Haiti 7
    # Mali 11
    # Kenya 9
    # PNG 20
    p1 = pops[0]
    s1 = int(sizes[0])
    p2 = pops[1]
    s2 = int(sizes[1])
    dd = dadi.Misc.make_data_dict(infile)
    if fold:
        fs = dadi.Spectrum.from_data_dict(dd, pop_ids=[p1, p2],
                                          projections=[s1, s2],
                                          polarized=False)
        if mask:
            fs.mask[1, :] = True
            fs.mask[:, 1] = True
        dadi.Plotting.plot_single_2d_sfs(fs, vmin=0.1)
    else:
        fs = dadi.Spectrum.from_data_dict(dd, pop_ids=[p1, p2],
                                          projections=[s1, s2], polarized=True)
        if mask:
            fs.mask[1, :] = True
            fs.mask[:, 1] = True
        dadi.Plotting.plot_single_2d_sfs(fs, vmin=0.1)
    print("Data Loaded")
    return(fs, dd)


def model_select(fs, pts_l):
    """Lists available models from demographic_models module. Also allows to
    add parameters on the command line directly or import from a config file.

    Parameters
    ------
    params: list, parameter names
    upper: list, upper bound
    lower: list, lower bound
    p0: list, initial guess

    Returns
    ------
    dadi_cmd: str
    """
    print("Available models: {}".format(help(demographic_models)))
    modeln = raw_input("Model name: ")
    model = getattr(__import__('demographic_models'), modeln)
    print("{}".format(help(model)))
    params = []
    p0 = []
    lower = []
    upper = []
    if os.path.exists("dadi.{}.config".format(modeln)):
        tf = raw_input("load default file?: ")
        if tf == "y":
            filein = "dadi.{}.config".format(modeln)
        else:
            filein = raw_input("path to config file: ")
    else:
        filein = raw_input("path to config file: ")
    if filein:
        p = []
        with open(filein, 'r') as parm:
            for line in parm:
                if not line.startswith("#"):
                    p.append(line.strip().split(","))
                else:
                    pms = line.strip().split()
        params, upper, lower = p
    else:
        pms = ''
        fin = ''
        while fin != "Done":
            params.append(raw_input("Add parameter "))
            lower.append(raw_input("lower bound "))
            upper.append(raw_input("upper bound "))
            fin = raw_input("Type 'Done' if finished ")
    params = map(float, params)
    p0 = params
    upper = map(float, upper)
    lower = map(float, lower)
    assert len(params) == len(p0)
    assert len(p0) == len(upper)
    assert len(upper) == len(lower)
    print("Parameters: {}".format(pms))
    print("Intial guess: {}".format(params))
    print("Upper bound: {}".format(upper))
    print("Lower bound: {}".format(lower))
    samplesize1 = raw_input("Sample size 1: ")
    samplesize2 = raw_input("Sample size 2: ")
    ns = map(int, (samplesize1, samplesize2))
    model2, model_ex, popt, ll, theta0 = run_dadisim(fs, model, p0, upper,
                                                     lower, ns, pts_l, pms)
    return(model2, model_ex, popt, ll, theta0)


def run_dadisim(fs, model, p0, upper, lower, ns, pts_l, pms, alg=2, maxit=100):
    """Presents options for the pre-loaded dadi demographic models. These
    are simple cases only. If need add your own to the demographic models
    module
    """
    model_ex = dadi.Numerics.make_extrap_log_func(model)
    ll_model = []
    theta = []
    p0opt = []
    runs = int(raw_input("how many runs?: "))
    r = 0
    while r < runs:
        print("Begin optimization")
        # perturb
        p0p = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper,
                                       lower_bound=lower)
        if alg == 1:
            print("optimize")
            popt = dadi.Inference.optimize(p0p, fs, model_ex, pts_l,
                                           lower_bound=lower,
                                           upper_bound=upper,
                                           verbose=len(p0p),
                                           maxiter=maxit)
        elif alg == 2:
            print("optimize_log")
            popt = dadi.Inference.optimize_log(p0p, fs, model_ex, pts_l,
                                               lower_bound=lower,
                                               upper_bound=upper,
                                               verbose=len(p0p),
                                               maxiter=maxit)
        elif alg == 3:
            print("optimize_lbfgsb")
            popt = dadi.Inference.optimize_lbfgsb(p0p, fs, model_ex, pts_l,
                                                  lower_bound=lower,
                                                  upper_bound=upper,
                                                  verbose=len(p0p),
                                                  maxiter=maxit)
        elif alg == 4:
            print("optimize_log_lbfgsb")
            popt = dadi.Inference.optimize_log_lbfgsb(p0p, fs, model_ex, pts_l,
                                                      lower_bound=lower,
                                                      upper_bound=upper,
                                                      verbose=len(p0p),
                                                      maxiter=maxit)
        elif alg == 5:
            print("optimize_log_fmin")
            popt = dadi.Inference.optimize_log_fmin(p0p, fs, model_ex, pts_l,
                                                    lower_bound=lower,
                                                    upper_bound=upper,
                                                    verbose=len(p0p),
                                                    maxiter=maxit)
        r += 1
        print("completing run {}".format(r))
        p0opt.append(popt)
        print("Optimization complete")
        print(pms)
        print("Best-fit parameters: {}".format(popt))
        model2 = model_ex(popt, ns, pts_l)
        ll_model.append(dadi.Inference.ll_multinom(model2, fs))
        print("Max log composite likelihood: {0}".format(ll_model))
        theta0 = dadi.Inference.optimal_sfs_scaling(model2, fs)
        theta.append(theta0)
        print("Optimal theta0, anc: {}".format(theta0))
        iix_ll = ll_model.index(max(ll_model))
        p0 = p0opt[iix_ll]
    iix_ll = ll_model.index(max(ll_model))
    print("Max ll: {}\t popt: {}\n".format(max(ll_model), p0opt[iix_ll]))
    return(model2, model_ex, p0opt[iix_ll], ll_model[iix_ll], theta[iix_ll])


def dadi_bs(dd, blocksize, totalsize, number_bs, pops, sizes, fold=True,
            mask=True):
    """Chunks the datadict constructed from dadi.Misc.make_data_dict(infile)
    Then reconstructs a new data_dict from chunked data and builds a fs object.
    The fs objects are then resampled with replacement to build the bootstrap.
    Blocks are resampled until totalsize. For example, 80Mb genome in 2Mb
    chunks will result in 40 blocks of 2Mb. These are then resampled back to
    a size of 80Mb by selection 40 blocks from the data_dict with replacement

    Parameters
    ------
    infile: file
    blocksize: int, size of contigous blocks, i.e., 2000000
    totalsize: int, how many of blocksize to combine
    number_bs: int, how many random fs

    Returns
    ------
    fs

    """
    p1 = pops[0]
    s1 = int(sizes[0])
    p2 = pops[1]
    s2 = int(sizes[1])
    # rebuild dict from dadi
    bsdict = defaultdict(list)
    for key in dd.keys():
        chrom = "_".join(key.split("_")[0:3])
        pos = int(key.split("_")[-1])
        bsdict[chrom].append(pos)
    # make blocks ids for parsing dd
    fsblock = []
    fsblock_id = []
    for key in bsdict.keys():
        v = np.array(sorted(bsdict[key]))
        b = 0
        s = 0
        e = blocksize
        chunks = max(v)/blocksize
        while b < chunks:  # number of chunks
            vs = v[np.where(np.logical_and(v >= s, v <= e))]
            fsblock.append([vs])
            fsblock_id.append(key)
            s += blocksize
            e += blocksize
            b += 1
    # make chunked fs dict from dd and dd_ids
    chunked_fs = {}
    for i, ids in enumerate(fsblock_id):
        new_keys = ["{}_{}".format(ids, k) for k in fsblock[i][0]]
        dd_bs = {nk: dd[nk] for nk in new_keys}
        if fold:
            fs = dadi.Spectrum.from_data_dict(dd_bs, pop_ids=[p1, p2],
                                              projections=[s1, s2],
                                              polarized=False)
            if mask:
                fs.mask[1, :] = True
                fs.mask[:, 1] = True
        else:
            fs = dadi.Spectrum.from_data_dict(dd_bs, pop_ids=[p1, p2],
                                              projections=[s1, s2],
                                              polarized=True)
            if mask:
                fs.mask[1, :] = True
                fs.mask[:, 1] = True
        chunked_fs["{}".format(i)] = fs
    # write bootstraps
    directory = "./bootstraps"
    if not os.path.exists(directory):
        os.makedirs(directory)
    nb = 0
    print("Creating Bootstraps")
    while nb < number_bs:
        # resample from fs dictionary
        fs = chunked_fs[random.choice(chunked_fs.keys())]
        while i < (totalsize/blocksize):
            fs += chunked_fs[random.choice(chunked_fs.keys())]
            i += 1
        nb += 1
        # write to file
        fs.to_file("bootstraps/{0:02d}.fs".format(nb))
    return(None)


def run_uncertainty(nboots, pts_l, popt, fs, model):
    """
    """
    # uncertainty estimates
    all_boot = [dadi.Spectrum.from_file('bootstraps/{0:02d}.fs'.format(ii))
                for ii in range(1, nboots + 1)]
    uncerts = dadi.Godambe.GIM_uncert(model, pts_l, all_boot, popt, fs,
                                      multinom=True)
#    uncerts_folded = dadi.Godambe.GIM_uncert(model, pts_l, all_boot,
#                                              popt_fold, fs.fold(),
#                                              multinom=True)

    print('\nEstimated parameter standard deviations from GIM: {0}'.
          format(uncerts))
    return(uncerts, all_boot)


def lrt_test(ll_lrt, p_lrt, nullparams_lrt, pts_l, fs, model, allboot):
    """Simple likelihood ratio test between nested models. Run simple model
    1st, so that model_ex is passed from more complex model

    Parameters
    ------
    ll_models: list, model likelihoods
    popts: list, optimized parameters
    nested_param: int, index of nested parameter to be 0

    Returns
    ------
    pval: float, pvalue for rejecting model1

    """
    pvalue = []
    for i in range(len(ll_lrt)-1):
        adj = dadi.Godambe.LRT_adjust(model, pts_l, allboot, p_lrt[i], fs,
                                      nested_indices=nullparams_lrt[i],
                                      multinom=True)
        D_adj = adj*2*(ll_lrt[-1] - ll_lrt[i])
        pval = dadi.Godambe.sum_chi2_ppf(D_adj, weights=(0.5, 0.5))
        print('p-value for rejecting model {0}: {1}'.format(i, pval))
        pvalue.append(pval)
    return(pvalue)


if __name__ == "__main__":
    infile = args.infile
    pops = args.pop
    sizes = args.size
    pts_l = [40, 50, 60]
    nboots = args.boots  # 100
    chunksize = args.chunksize  # 2E6
    totalsize = args.totalsize  # 80E6
    fs, dd = load_data(infile, pops, sizes)
    if args.lrt:
        nestedmodels = raw_input("number nested: ")
        assert int(nestedmodels) > 1
        b = 0
        p_lrt = []
        ll_lrt = []
        nullparams_lrt = []
        while b < int(nestedmodels):
            nparams = raw_input("index of null params: ")
            model, model_ex, popt, ll, theta = model_select(fs, pts_l)
            dadi_bs(dd, chunksize, totalsize, nboots, pops, sizes)
            uncert, allboot = run_uncertainty(nboots, pts_l, popt, fs,
                                              model_ex)
            lun = [(o + p, o, o - p) for p, o in zip(uncert, popt)]
            lun.append((uncert[-1]-theta, theta, uncert[-1]+theta))
            print(lun)
            p_lrt.append(popt)
            ll_lrt.append(ll)
            nullparams_lrt.append(map(int, nparams.split(",")))
            b += 1
            pylab.figure(1)
            dadi.Plotting.plot_2d_comp_multinom(model, fs, vmin=1,
                                                resid_range=3,
                                                pop_ids=(pops[0], pops[1]))
        pvalue = lrt_test(ll_lrt, p_lrt, nullparams_lrt, pts_l, fs,
                          model_ex, allboot)
        print(p_lrt)
        print(ll_lrt)
    else:
        model, model_ex, popt, ll, theta = model_select(fs, pts_l)
        dadi_bs(dd, chunksize, totalsize, nboots, pops, sizes)
        uncert, allboot = run_uncertainty(nboots, pts_l, popt, fs, model_ex)
        lun = [(o - p, o, o + p) for p, o in zip(uncert, popt)]
        lun.append((uncert[-1]-theta, theta, uncert[-1]+theta))
        print(lun)
        # Plot a comparison of the resulting fs with the data.
        pylab.figure(1)
        dadi.Plotting.plot_2d_comp_multinom(model, fs, vmin=1, resid_range=3,
                                            pop_ids=(pops[0], pops[1]))
