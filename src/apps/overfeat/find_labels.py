import ast
import glob
from shutil import copyfile, rmtree
import os
from os import chdir, mkdir

infile = 'data.txt'

dir_black = 'n09636339'
dir_white = 'n09638875'

chdir(dir_black)
files_black = sorted(glob.glob('*.JPEG'))
chdir('../'+dir_white)
files_white = sorted(glob.glob('*.JPEG'))
chdir('..')

targets_underwear = set(['brassiere', 'maillot', 'miniskirt'])
targets_bullet = set(['bulletproof vest'])
targets_mask = set(['mask', 'ski mask'])
targets_monkey = set(['gorilla', 'orangutan', 'chimpanzee'])

target_dir_underwear = 'offensive_underwear'
target_dir_bullet = 'offensive_bullet'
target_dir_mask = 'offensive_mask'
target_dir_monkey = 'offensive_monkey'

target_dirs = [target_dir_underwear, target_dir_bullet, target_dir_mask, target_dir_monkey]

for t_dir in target_dirs:
    if os.path.exists(t_dir):
        rmtree(t_dir)
        mkdir(t_dir)
    else:
        mkdir(t_dir)
    

with open(infile) as inf:
    inf.readline()
    
    lines = map(lambda l: l.split('\t'), inf.readlines())
    lines = [(race, set(ast.literal_eval(l))) for (race,l) in lines]
    
    blacks_labels = [s for (race, s) in lines if race == 'Black']
    whites_labels = [s for (race, s) in lines if race == 'White']
    
    assert len(blacks_labels) == len(files_black)
    assert len(whites_labels) == len(files_white)
    
    blacks = zip(blacks_labels, files_black)
    whites = zip(whites_labels, files_white)
    
    off_black_under = [f for (s, f) in blacks if s.intersection(targets_underwear)]
    off_black_bullet = [f for (s, f) in blacks if s.intersection(targets_bullet)]
    off_black_mask = [f for (s, f) in blacks if s.intersection(targets_mask)]
    off_black_monkey = [f for (s, f) in blacks if s.intersection(targets_monkey)]
    off_white_under = [f for (s, f) in whites if s.intersection(targets_underwear)]
    off_white_bullet = [f for (s, f) in whites if s.intersection(targets_bullet)]
    off_white_mask = [f for (s, f) in whites if s.intersection(targets_mask)]
    off_white_monkey = [f for (s, f) in whites if s.intersection(targets_monkey)]

    for f in off_black_under:
        copyfile(dir_black+'/'+f, target_dir_underwear+'/'+f)
        
    for f in off_black_bullet:
        copyfile(dir_black+'/'+f, target_dir_bullet+'/'+f)
        
    for f in off_black_mask:
        copyfile(dir_black+'/'+f, target_dir_mask+'/'+f)
        
    for f in off_black_monkey:
        copyfile(dir_black+'/'+f, target_dir_monkey+'/'+f)
    

    for f in off_white_under:
        copyfile(dir_white+'/'+f, target_dir_underwear+'/'+f)
        
    for f in off_white_bullet:
        copyfile(dir_white+'/'+f, target_dir_bullet+'/'+f)
        
    for f in off_white_mask:
        copyfile(dir_white+'/'+f, target_dir_mask+'/'+f)
        
    for f in off_white_monkey:
        copyfile(dir_white+'/'+f, target_dir_monkey+'/'+f)
    
    