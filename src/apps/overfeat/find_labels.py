import ast
import glob
from shutil import copyfile, rmtree
import os
from os import chdir, mkdir
import sys


def main(argv=sys.argv):
    if len(argv) != 5:
        usage(argv)


    dir_black = argv[1]
    dir_white = argv[2]
    infile = argv[3]
    label = argv[4]

    files_black = sorted(glob.glob('{}/*.JPEG'.format(dir_black)))
    files_white = sorted(glob.glob('{}/*.JPEG'.format(dir_white)))

    target_dir = "images_{}".format(label)
    if os.path.exists(target_dir):
        rmtree(target_dir)

    mkdir(target_dir)
        

    with open(infile) as inf:
        inf.readline()
        lines = map(lambda l: l.split('\t'), inf.readlines())
        lines = [(fname, set(ast.literal_eval(l))) for (fname, l) in lines]

        blacks_labels = [s for (race, s) in lines if race == 'Black']
        whites_labels = [s for (race, s) in lines if race == 'White']

        assert len(blacks_labels) == len(files_black)
        assert len(whites_labels) == len(files_white)

        blacks_labels = zip(files_black, blacks_labels)
        whites_labels = zip(files_white, whites_labels)

        for labels in [blacks_labels, whites_labels]:
            if 'not-' in label:
                neg_label = label[4:]
                targets = [fname for (fname, l) in labels if not neg_label in l]
            else:
                targets = [fname for (fname, l) in labels if label in l]
      
            for f in targets:
                copyfile(f, target_dir+'/'+os.path.basename(f))


def usage(argv):
    print "Usage:%s <black_dir> <white_dir> <data.txt> <(not-?)label>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main()) 
    
    