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


def msp2dadi(tree_sequence, npops):
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
    fsdict = {}
    dadidict = {}
    pix = []
    # store replicates as list
    tslist = []
    for ts in tree_sequence:
        tslist.append(ts)
    # set pop index
    pix = [tslist[0].get_samples(pop) for pop in range(npops)]
    iterations = len(tslist)
    for i, p in enumerate(pix):
        fsdict[i] = np.zeros((iterations, len(p)))
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
                # pos is unused in sfs, left in code for future
                pos = allel.SortedIndex([int(v.position) for v in ts.variants()])
                for i, pop in enumerate(pix):
                    gtpop = gt[:, pop]
                    try:
                        sfs = allel.sfs(gtpop.count_alleles()[:, 1])
                        # sfs = np.pad(sfs, (0, len(pix - len(sfs)), mode='constant')
                        sfs.resize(len(pop), refcheck=False)
                        fsdict[i][r] = fsdict[i][r] + sfs
                    except IndexError:
                        print("insufficient mutations in population")
    # get mean for each pop and import to dadi
    for pop in fsdict.keys():
        fs = np.sum(np.array(fsdict[pop]), axis=0)
        dadidict[pop] = dadi.Spectrum(fs, pop_ids=["pop{}".format(pop)])
    return(dadidict)


##Example of msprime with 3 populations
    # 2 join events
#    dem_list=[msp.MassMigration(time=10, source=0, destination=1, proportion=1.0),
#              msp.MassMigration(time=100, source=1, destination=2, proportion=1.0)]
#    # 3 populations
#    popcfg = [msp.PopulationConfiguration(sample_size=10, initial_size=100),
#              msp.PopulationConfiguration(sample_size=10, initial_size=1000),
#              msp.PopulationConfiguration(sample_size=10, initial_size=500)]
#    # msprime simulate
#    tree_sequence = msp.simulate(population_configurations=popcfg,
#                                 Ne=100000,
#                                 length=1e5,
#                                 recombination_rate=2e-8,
#                                 mutation_rate=2e-6,
#                                 demographic_events=dem_list,
#                                 num_replicates=1000)
#    # run fx and return sfs as dadi format
#    print("Length in bp: {}".format(length*num_replicates))
#    dadidict = msp2dadi(tree_sequence, 3)
