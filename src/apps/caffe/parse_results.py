#!/usr/bin/python

import os, sys
import random

infile = 'results_black.txt'
infile2 = 'results_white.txt'
outfile = 'data.txt'


with open(outfile,"w") as outf:
    outf.write('Race\tLabels\n')
    
    for line in open(infile).readlines():
        labels = line.split('\t')[1].strip()
        outf.write('Black\t{}\n'.format(labels))
    
        
    for line in open(infile2).readlines():
        labels = line.split('\t')[1].strip()
        outf.write('White\t{}\n'.format(labels))