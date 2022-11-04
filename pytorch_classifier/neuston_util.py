#!/usr/bin/env
"""This module exists to run useful auxiliary neuston tasks"""
import argparse
import os
import csv

import numpy as np

from neuston_data import HerringrunnerDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def calc_img_norm(args):

    tforms=transforms.Compose([transforms.Resize(2*[args.resize]),transforms.ToTensor()])

    with open(args.SRC) as f:
        filelist = [line.strip() for line in f.readlines()]
    dataset = HerringrunnerDataset(filelist, resize=args.resize)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_batches = len(dataloader)

    pop_mean = []
    pop_std0 = []
    for i,data in enumerate(dataloader,1):
        img_data,_,_ = data
        # shape (batch_size, 3, height, width)
        numpy_image = img_data.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        #batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)

        if i%100==0:
            line = '\n{:.1f}% ({} of {}) MEAN={} STD={}'
            line = line.format(100*i/num_batches,i,num_batches,
                               np.array(pop_mean).mean(axis=0)[0],
                               np.array(pop_std0).mean(axis=0)[0])
            print(line)
        else:
            print('.',end='',flush=True)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    mean = np.array(pop_mean).mean(axis=0)
    std0 = np.array(pop_std0).mean(axis=0)
    return mean,std0


def main(args):
    if args.cmd=='CALC_IMG_NORM':
        print('Calculating Image Normalization MEAN and STD...')
        mean,std = calc_img_norm(args)
        print('MEAN={}, STD={}'.format(mean,std))
    else:
        raise InputError('args.cmd not recognized:', args.cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', help='These sub-commands are mutually exclusive.')

    # IMAGE NORMALIZATION
    imgnorm = subparsers.add_parser('CALC_IMG_NORM', help='Calculate the MEAN and STD of dataset for image normalizing')
    imgnorm.add_argument('SRC')
    imgnorm.add_argument('--resize', metavar='N', default=299, type=int, choices=[224,299], help='Default is 299 (for inception_v3)')
    imgnorm.add_argument('--batch-size', metavar='B', default=108, help='Number of images per minibatch')

    # run util command
    args = parser.parse_args()
    main(args)

