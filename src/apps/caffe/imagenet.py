import subprocess
from scipy.misc import imread, imsave, imresize
import sys
import numpy as np
import pandas as pd
import os


def color_type(f_name):
    image = imread(f_name)
    if(len(image.shape)<3):
          return 'gray'
    elif len(image.shape)==3:
          return 'Color(RGB)'
    else:
          return 'others'

BATCH_SIZE = 20

def main(argv=sys.argv):
    if len(argv) != 3:
        usage(argv)

    image_dir = argv[1]
    output_file = argv[2]
    ret = subprocess.call("rm -rf _temp", shell=True)
    ret |= subprocess.call("mkdir -p _temp", shell=True)
    assert ret == 0

    import glob
    filenames = glob.glob(image_dir+'/*')


    num_batches = (len(filenames)+BATCH_SIZE-1)/BATCH_SIZE
    print 'running {} batches of {} images ({} images total)...'.format(num_batches, BATCH_SIZE, len(filenames))

    with open(output_file,"w") as outf:
        for i in range(num_batches):
            print "STARTING BATCH {}".format(i+1)
            for filename in filenames[i*BATCH_SIZE:(i+1)*BATCH_SIZE]:
                if color_type(filename) == 'gray':
                    print 'gray image ignored: {}'.format(filename)
                else:
                	ret = subprocess.call("echo {} >> _temp/input_{}.txt".format(filename, i), shell=True)
                	assert ret == 0

            ret = subprocess.call("../python/detect.py --crop_mode=selective_search "
                "--pretrained_model=../models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel "
                "--model_def=../models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt "
                "--raw_scale=255 "
                "_temp/input_{}.txt ".format(i) +
                "_temp/det_output.h5", shell=True, stderr=sys.stdout.fileno())
            assert ret == 0, "return code was {}".format(ret)

            temp = pd.read_hdf('_temp/det_output.h5', 'df')
            dfs = {}
            for (name, df) in temp.reset_index().groupby('filename'):
                dfs[name] = df

            with open('../data/ilsvrc12/det_synset_words.txt') as f:
                labels_df = pd.DataFrame([
                    {
                        'synset_id': l.strip().split(' ')[0],
                        'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                    }
                    for l in f.readlines()
                ])
            labels_df.sort('synset_id')

            predictions = {}
            for (name, df) in dfs.iteritems():
                predictions[name] = pd.DataFrame(np.vstack(df.prediction.values), columns=labels_df['name'])

            
            for (name, predictions_df) in predictions.iteritems():
                max_s = predictions_df.max(0)
                max_s.sort(ascending=False)
                max_predictions = max_s[:5]
                #subprocess.call("echo '{}' '{}' >> {}".format(os.path.basename(name), max_predictions.index.tolist(), output_file), shell=True)
                outf.write('{}\t{}\n'.format(os.path.basename(name), max_predictions.index.tolist()))
            outf.flush()

    subprocess.call("rm -rf _temp", shell=True)

def usage(argv):
    print "Usage:%s <image_dir> <output_file>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())