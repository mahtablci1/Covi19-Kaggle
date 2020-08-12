#!/usr/bin/env python

'''


Running time: 


'''

import argparse
import os
import fasttext

import utils

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def main(args):

    config = utils.read_config()
    utils.check_valid_dataset(config,args)
    input_root = config['datasets'][args.dataset]['input_dir']

    input_dir = os.path.join(input_root,'fasttext')
    dataset_path = os.path.join(input_dir,'train.txt')
    training_sample_path = os.path.join(input_dir,'train_sample.txt')
    output_dir = input_dir

    valid_path = os.path.join(input_dir,'valid.txt')

    os.makedirs(output_dir,exist_ok=True)
    model = fasttext.train_supervised(
        dataset_path,
        thread=args.njobs,
        autotuneValidationFile=valid_path,
        verbose=3,
        autotuneDuration=60*60,
        # Do not search these:
        # epoch=25,
        # lr=1.0,
        wordNgrams=1,
        # dim=100,
        # minCount=1,
        minn=0,
        maxn=0,
        # bucket=2000000,
        loss='hs', # Go very fast!
        )

    path = os.path.join(output_dir,'model.ftz')
    model.save_model(path)

    print('Training set metrics:')
    print_results(*model.test(training_sample_path))

    print('Validation set metrics:')
    print_results(*model.test(valid_path))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset',type=str,required=True,help='name of dataset')
    parser.add_argument('-j','--njobs',type=int,required=False,default=1,help='number of CPU cores to train')
    args = parser.parse_args()
    main(args)
