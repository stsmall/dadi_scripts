#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
vcf2dadi.py
python vcf2dadi.py -p1 -p2 -p3 -ped -vcf -f1 -f2
Created on Mon Jul 17 16:24:45 2017
@author: scott
"""
import argparse
from collections import defaultdict
import re

parser = argparse.ArgumentParser()
parser.add_argument('-in', "--ingroup", type=str, required=True,
                    help="name of ingroup")
parser.add_argument('-out', "--outgroup", type=str, required=True,
                    help="name of outgroup")
parser.add_argument('-ped', "--pedfile", type=str, required=True,
                    help="path to pedfile with pop and individuals")
parser.add_argument('-vcf', "--vcf_in", type=str, required=True,
                    help="path to vcf")
parser.add_argument('-f1', "--fasta1", type=str, required=False,
                    help="path to population ref fasta")
parser.add_argument('-f2', "--fasta2", type=str, required=False,
                    help="path to population outgroup fasta")
args = parser.parse_args()


def parse_vcf(vcfin, popinfo, ingroup, outgroup):
    """Requires vcf and pop strings along with indiviual info to make a dict
    object containing counts of alleles
    """
    # parse ped
    peddict = defaultdict(list)
    with open(popinfo, 'r') as ped:
        for line in ped:
            if line.strip():
                x = line.strip().split()
                peddict[x[0]].append(x[1])
            else:
                continue
    poplist = peddict.keys()
    # open file
    dadi = open("dadi.notriplets.in", 'w')
    dadi.write("{}\t{}\tAllele1\t".format(ingroup, outgroup))
    for pop in poplist:
        dadi.write("{}\t".format(pop))
    dadi.write("Allele2\t")
    for pop in poplist:
        dadi.write("{}\t".format(pop))
    dadi.write("Gene\tPosition\n")
    # parse vcf
    with open(vcfin, 'r') as vcf:
        for line in vcf:
            if not line.startswith("##"):
                if line.startswith("#CHROM"):
                    pop_iix = line.strip().split()
                else:
                    x = line.strip().split()
                    chrom = x[0]
                    pos = x[1]
                    ra = x[3]
                    aa = x[4]
                    dadi.write("-{}-\t-{}-\t{}\t".format(ra, aa, ra))
                    for pop in poplist:
                        rac = 0
                        for sample in peddict[pop]:
                            p = pop_iix.index(sample)
                            rac += x[p][0].count("0")
                        dadi.write("{}\t".format(rac))
                    dadi.write("{}\t".format(aa))
                    for pop in poplist:
                        aac = 0
                        for sample in peddict[pop]:
                            p = pop_iix.index(sample)
                            aac += x[p][0].count("1")
                        dadi.write("{}\t".format(aac))
                    dadi.write("{}\t{}\n".format(chrom, pos))
    dadi.close()
    return(None)


def parse_fasta(fasta_ref, fasta_out, ingroup):
    """takes triplet info from passed fasta
    """
    rdict = {}
    odict = {}
    # load ref fasta
    f = ''
    with open(fasta_ref, 'r') as fasta:
        for line in fasta:
            if line.startswith(">"):
                chrom = line.strip().lsplit(">")
                line = next(fasta)
                # line = fasta.next()
                while not line.startswith(">"):
                    f += line.strip()
                    line = next(fasta)
                    # line = fasta.next()
                rdict[chrom] = f
                f = ''
    # load outgroup fasta
    r = ''
    with open(fasta_ref, 'r') as fasta:
        for line in fasta:
            if line.startswith(">"):
                chrom = line.strip().lsplit(">")
                line = next(fasta)
                # line = fasta.next()
                while not line.startswith(">"):
                    r += line.strip()
                    line = next(fasta)
                    # line = fasta.next()
                odict[chrom] = r
                r = ''
    # get first chrom
    with open("dadi.notriplets.in", 'r') as dadi:
        for line in dadi:
            if line.startswith(ingroup):
                line = next(dadi)
                # line = dadi.next()
                x = line.strip().split()
                chrom = x[-2]
    # remake dadi infile
    dtrip = open("dadi.triplets.in", 'w')
    with open("dadi.notriplets.in", 'r') as dadi:
        for line in dadi:
            if line.startswith(ingroup):
                dtrip.write(line)
            else:
                x = line.strip().split()
                if chrom != x[-2]:
                    iters = re.finiter(r"[ATCG]", rdict[chrom])
                    indices = [m.start(0) for m in iters]
                chrom = x[-2]
                pos_v = int(x[-1]) - 1  # vcf is not 0 based
                pos_a = indices[pos_v]
                try:
                    # verify that the ref sites are the same
                    if x[0][1] != rdict[chrom][pos_a]:
                        raise Exception("Ref sequence and Ref do not match")
                    else:
                        x[0] = rdict[chrom][pos_a-1:pos_a+2]
                    if rdict[chrom][pos_a] == '-':
                        raise Exception("outgroup sequence is gap")
                    else:
                        x[1] = odict[chrom][pos_a-1:pos_a+2]
                    dtrip.write("{}\n".format("\t".join(x)))
                except:
                    pass  # this will not write the position
    dtrip.close()
    return(None)


if __name__ == "__main__":
    vcfin = args.vcf_in
    popinfo = args.pedfile
    ingroup = args.ingroup
    # check for fasta
    if args.fasta1 and args.fasta2 is None:
        parser.error("--fasta1 requires --fasta2")
    elif args.fasta2 and args.fasta1 is None:
        parser.error("--fasta2 requires --fasta1")
    else:
        fasta_ref = args.fasta1
        fasta_out = args.fasta2
    # make dadi
    parse_vcf(vcfin, popinfo, ingroup, args.outgroup)
    # add triplet info
    if fasta_ref:
        fastadict = parse_fasta(fasta_ref, fasta_out, ingroup)
