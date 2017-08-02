#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:09:14 2017

@author: scott
"""
from numpy import array
import numpy as np
import dadi
import demographic_models
import pylab
import argparse
from collections import defaultdict
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--infile", type=str, required=True,
                    help='infile name')
parser.add_argument('-p', "--pop", nargs='+', required=True,
                    help='name pop1 pop2')
parser.add_argument('-s', "--size", nargs='+', required=True,
                    help='size pop1 pop2')
args = parser.parse_args()


def load_data(infile, pops, sizes, fold=True):
    """Read in the dadi formatted file creates a fs object

    Parameters
    ------
    infile: file, dadi formatted file
    fold: bool, True is folded, default

    Returns
    ------
    fs

    """
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
        dadi.Plotting.plot_single_2d_sfs(fs, vmin=0.1)
    else:
        fs = dadi.Spectrum.from_data_dict(dd, pop_ids=[p1, p2],
                                          projections=[s1, s2], polarized=True)
        dadi.Plotting.plot_single_2d_sfs(fs, vmin=0.1)
    print("Data Loaded")
    return(fs)


def model_select(fs, pts_l):
    """
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
    model2, model_ex, popt = run_dadisim(fs, model, p0, upper, lower, ns,
                                         pts_l)
    return(model2, model_ex, popt)


def run_dadisim(fs, model, p0, upper, lower, ns, pts_l, alg=2, maxit=100):
    """Runs the specified dadi model
    """
    model_ex = dadi.Numerics.make_extrap_log_func(model)
    ll_model = []
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
        print("Best-fit parameters: {}".format(popt))
        model2 = model_ex(popt, ns, pts_l)
        ll_model.append(dadi.Inference.ll_multinom(model2, fs))
        print("Max log composite likelihood: {0}".format(ll_model))
        theta0 = dadi.Inference.optimal_sfs_scaling(model2, fs)
        print("Optimal theta0, anc: {}".format(theta0))
        print("Ne = {}".format(theta0/(4*4E7*2.9E-9)))
        iix_ll = ll_model.index(max(ll_model))
        p0 = p0opt[iix_ll]
    iix_ll = ll_model.index(max(ll_model))
    print("Max ll: {}\t popt: {}\n".format(max(ll_model), p0opt[iix_ll]))
    return(model2, model_ex, p0opt[iix_ll])


def dadi_bs(infile, blocksize, totalsize, number_bs, pops, sizes, fold=True):
    """makes bootstrap fs from input dadi file

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
    dd = dadi.Misc.make_data_dict(infile)
    for key in dd.keys():
        chrom = "_".join(key.split("_")[0:3])
        pos = int(key.split("_")[-1])
        bsdict[chrom].append(pos)
    # make blocks
    fsblock = []
    fsblock_id = []
    for key in bsdict.keys():
        v = np.array(sorted(bsdict[key]))
        b = 0
        s = 0
        e = blocksize
        while b < max(v)/blocksize:
            vs = v[np.where(np.logical_and(v >= s, v <= e))]
            fsblock.append([vs])
            fsblock_id.append(key)
            s += e
            e += blocksize
            b += 1
    directory = "./bootstraps"
    if not os.path.exists(directory):
        os.makedirs(directory)
    nb = 0
    print("Creating Bootstraps")
    while nb < number_bs:
        fs_randkeys = []
        fs_rand = np.random.choice(range(0, len(fsblock_id)),
                                   int(totalsize/blocksize))
        for i in fs_rand:
            for j in fsblock[i]:
                newkey = ["{}_{}".format(fsblock_id[i], k) for k in j]
                fs_randkeys.append(newkey)
        fs_randkeys2 = [item for sublist in fs_randkeys for item in sublist]
        dd_bs = {newkey: dd[newkey] for newkey in fs_randkeys2}
        if fold:
            fs = dadi.Spectrum.from_data_dict(dd_bs, pop_ids=[p1, p2],
                                              projections=[s1, s2],
                                              polarized=False)
        else:
            fs = dadi.Spectrum.from_data_dict(dd_bs, pop_ids=[p1, p2],
                                              projections=[s1, s2],
                                              polarized=True)
        nb += 1
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
    return(uncerts)


if __name__ == "__main__":
    infile = args.infile
    pops = args.pop
    sizes = args.size
    pts_l = [40, 50, 60]
    nboots = 100
    chunksize = 2E6
    totalsize = 80E6
    fs = load_data(infile, pops, sizes)
    model, model_ex, popt = model_select(fs, pts_l)
    dadi_bs(infile, chunksize, totalsize, nboots, pops, sizes)
    run_uncertainty(nboots, pts_l, popt, fs, model_ex)
    # Plot a comparison of the resulting fs with the data.
    pylab.figure(1)
    dadi.Plotting.plot_2d_comp_multinom(model, fs, vmin=1, resid_range=3,
                                        pop_ids=(pops[0], pops[1]))
    pylab.savefig("{}_{}.png".format(pops[0], pops[1]), dpi=50)
