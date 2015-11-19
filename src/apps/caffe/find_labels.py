import ast
import glob
from shutil import copyfile, rmtree
import os
from os import chdir, mkdir
import sys


def main(argv=sys.argv):
    if len(argv) != 6:
        usage(argv)


    dir_black = argv[1]
    dir_white = argv[2]
    infile_black = argv[3]
    infile_white = argv[4]
    label = argv[5]

    target_dir = "images_{}".format(label)
    if os.path.exists(target_dir):
        rmtree(target_dir)

    mkdir(target_dir)
        

    for (infile, image_dir) in [(infile_black, dir_black), (infile_white, dir_white)]:
        with open(infile) as inf:
            lines = map(lambda l: l.split('\t'), inf.readlines())
            lines = [(fname, set(ast.literal_eval(l))) for (fname, l) in lines]

            if 'not-' in label:
                neg_label = label[4:]
                targets = [fname for (fname, l) in lines if not neg_label in l]
            else:
                targets = [fname for (fname, l) in lines if label in l]
      
            for f in targets:
                copyfile(image_dir+'/'+f, target_dir+'/'+f)

   

def usage(argv):
    print "Usage:%s <black_dir> <white_dir> <res_black> <res_white> <(not-?)label>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main()) 
    