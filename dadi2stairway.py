#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:40:20 2017

@author: scott
"""
import numpy as np
import dadi
from collections import defaultdict
from collection import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--infile", type=str, required=True,
                    help='infile name')
parser.add_argument('-p', "--pop", nargs='+', required=True,
                    help='name pop1 pop2')
parser.add_argument('-s', "--size", nargs='+', required=True,
                    help='pop1 pop2')
parser.add_argument('-m', "--missing", type=float, default=1.0,
                    help="remove sites with missing data in excess of size *2")
parser.add_argument('-a', "--ancestral", type=str, default=None,
                    help="file where ancestral sites are trusted")
parser.add_argument("--thin", action="store_true", help="remove sites within"
                    "1000bp of each other")
args = parser.parse_args()


def makedadidict(dadi_infile, pop, size, allele2):
    """
    """
    dadidict = defaultdict(list)
    with open(dadi_infile, 'r') as dad:
        for line in dad:
            if "Allele1" in line:
                continue
            else:
                dadlist = line.strip().split()
                p_iix = [p for p, x in enumerate(dadlist) if x == pop]
                sample_size = int(x[p_iix][0]) + int(x[p_iix][1])
                if sample_size == size:  # no missing
                    d = "{}\t{}\t{}\{}\t{}".format("\t".join(x[:3]),
                                                   x[p_iix][0],
                                                   x[allele2],
                                                   x[p_iix][1],
                                                   "\t".join(x[-2:]))
                    dadidict[":".join(x[-2:])].append(d)
    return(dadidict)


def ancfilter(anc_sites, dadidict):
    """
    """
    # filter by anc
    if anc_sites:
        anclist = []
        with open(anc_sites, 'r') as anc:
            for line in anc:
                x = line.strip().split()
                anclist.append("{}:{}".format(x[0], x[1]))
        ancarray = np.array(anclist)
        for site in dadidict.keys():
            if not np.any([site in aa for aa in ancarray]):
                del dadidict[site]
    return(dadidict)


def thinfilter(thin, dadidict):
    """
    """
    dadidictOrdered = OrderedDict(sorted(dadidict.items()))
    for site in dadidictOrdered.keys():
        first_chrom = site.split(":")[0]
        first_site = int(site.split(":")[1])
        break
    for site in dadidictOrdered.keys():
        if site.split(":")[0] == first_chrom:
            if int(site.split(":")[1]) < (first_site + thin):
                del dadidictOrdered[site]
            else:
                first_site = int(site.split(":")[1])
        else:
            first_chrom = site.split(":")[0]
            first_site = int(site.split(":")[1])
    return(dadidict)


def filterdadi_in(dadi_infile, poplist, sizes, miss, anc_sites, thin=None):
    """Remove sites from dadi if they fail the missing and ancestral criteria
    """
    size_miss = [i * miss for i in sizes]
    fsdict = {}
    with open(dadi_infile, 'r') as f:
        header = f.readline().strip().split()
    for i, pop in enumerate(poplist):
        f = open("filtered_dadi.{}.out".format(pop), 'w')
        f.write("{}\t{}\tAllele2\t{}\tGene\Position\n".format("\t".join(header[:3]), pop, pop))
        dadidict = makedadidict(dadi_infile, pop, size_miss[i],
                                header.index("Allele2"))
        # thin by anc
        if anc_sites:
            dadidict = ancfilter(anc_sites, dadidict)
        # thin by distance
        if thin:
            dadidict = thinfilter(thin, dadidict)
        # write to outfile
        for key in dadidict.keys():
            f.write("{}\n".format(dadidict[key]))
        f.close()
        # read file to sfs
        dd = dadi.Misc.make_data_dict("filtered_dadi.{}.out".format(pop))
        if anc_sites:
            sfs = dadi2sfs(dd, pop, size_miss[i], fold=False)
        fsdict[pop] = sfs
    return(fsdict)


def dadi2sfs(dd, pop, size, fold=True, mask=False):
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
    # sample sizes in diploid
    # Haiti 7
    # Mali 11
    # Kenya 9
    # PNG 20
    if fold:
        fs = dadi.Spectrum.from_data_dict(dd, pop_ids=[pop],
                                          projections=[size],
                                          polarized=False)
    else:
        fs = dadi.Spectrum.from_data_dict(dd, pop_ids=[pop],
                                          projections=[size],
                                          polarized=True)
    return(fs)


def writesfs2file(fsdict):
    """
    """
    with open("stairway_plot.fs", 'w') as sf:
        for pop in fsdict.keys():
            sf.write("{}\t{}\n".format(pop, fsdict[pop]))

    return(None)


if __name__ == "__main__":
    dadi_infile = args.infile
    poplist = args.pop
    sizes = args.size
    miss = args.missing
    anc = args.ancestral
    fsdict = filterdadi_in(dadi_infile, poplist, sizes, miss, anc, args.thin)
    writesfs2file(fsdict)
