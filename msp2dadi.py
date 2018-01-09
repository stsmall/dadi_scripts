#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:07:13 2017
@author: stsmall
"""

import numpy as np
import allel
import msprime as msp
import dadi


def msp2dadi(tree_sequence, npops, average=False, mask_corners=True):
    """SFS from msprime tree obj using scikit-allel

    Parameters
    ------
    tree_sequence: obj, obj from msprime.simulate()
    npops: int, number of populations simulated

    Returns
    ------
    dadidict: dict, dictionary of dadi formated frequency spectrums keys as
        populations

    """
    pix = []
    # store replicates as list
    tslist = []
    for ts in tree_sequence:
        tslist.append(ts)
    # set pop index
    pix = [tslist[0].get_samples(pop) for pop in range(npops)]
    iterations = len(tslist)
    dim = [iterations]
    [dim.append(len(p) + 1) for p in pix]
    fsdict = np.zeros(dim, np.int_)
    # get sfs from allel
    for r, ts in enumerate(tslist):
        muts = ts.get_num_mutations()
        if muts > 0:
            sample_size = ts.get_sample_size()
            if muts > 0:
                V = np.zeros((muts, sample_size), dtype=np.int8)
                for variant in ts.variants():
                    V[variant.index] = variant.genotypes
                gt = allel.HaplotypeArray(V)
                # pos is unused in sfs, left in code for future use
                pos = allel.SortedIndex([int(v.position) for v in ts.variants()])
                aclist = []
                for p in pix:
                    aclist.append(gt[:, p].count_alleles()[:, 1])
                if npops == 1:
                    ac = aclist[0]
                    unique, counts = np.unique(ac, return_counts=True)
                    for k in dict(zip(unique, counts)).items():
                        fsdict[r][k[0]] = k[1]
                elif npops == 2:
                    for i, j in zip(*aclist):
                        fsdict[r][i, j] += 1
                elif npops == 3:
                    for i, j, k in zip(*aclist):
                        fsdict[r][i, j, k] += 1
                elif npops == 4:
                    for i, j, k, m in zip(*aclist):
                        fsdict[r][i, j, k, m] += 1
                elif npops == 5:
                    for i, j, k, m, n in zip(*aclist):
                        fsdict[r][i, j, k, m, n] += 1
                elif npops == 6:
                    for i, j, k, m, n, p in zip(*aclist):
                        fsdict[r][i, j, k, m, n, p] += 1
    if average:
        data = np.average(fsdict, axis=0)
    else:
        data = np.sum(fsdict, axis=0)
    pids = ["pop{}".format(p) for p in range(npops)]
    SF = dadi.Spectrum(data, pop_ids=pids, mask_corners=mask_corners)
    return(SF)


# Example of msprime with 3 populations
    # 2 join events
    dem_list = [msp.MassMigration(time=100, source=0, destination=1, proportion=1.0),
                msp.MassMigration(time=100, source=0, destination=1, proportion=1.0)]
    # 3 populations
    popcfg = [msp.PopulationConfiguration(sample_size=10, initial_size=100),
              msp.PopulationConfiguration(sample_size=10, initial_size=1000),
              msp.PopulationConfiguration(sample_size=10, initial_size=100)]
    # msprime simulate
    tree_sequence = msp.simulate(population_configurations=popcfg,
                                 Ne=100000,
                                 length=2e6,
                                 recombination_rate=2e-8,
                                 mutation_rate=2e-8,
                                 demographic_events=dem_list,
                                 num_replicates=10)
    # run fx and return sfs as dadi format
    # print("Length in bp: {}".format(length*num_replicates))
    msp2dadi(tree_sequence, 3)
