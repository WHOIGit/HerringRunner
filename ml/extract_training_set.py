#!/usr/bin/env python3
import argparse
import collections
import os
import pickle
import random

import cv2


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int,
    help='maximum number of samples to extract')
parser.add_argument('datafile', type=str,
    help='path to a pickled data file')
args = parser.parse_args()


# Data type in the pickled file, from prepare_fast_load.py
Frame = collections.namedtuple('Frame', ('image', 'fish'))
with open(args.datafile, 'rb') as f:
    frames = pickle.load(f)
assert len(frames) > 0


frames_with_fish = set()
frames_without_fish = set()
for framenum, frame in frames.items():
    (frames_with_fish if frame.fish else frames_without_fish).add(framenum)


count = min(len(frames_with_fish), len(frames_without_fish))
if args.n:
    count = min(count, args.n)
print('Extracting', count, 'frames of each kind')
frames_with_fish = random.sample(frames_with_fish, count)
frames_without_fish = random.sample(frames_without_fish, count)


os.makedirs('training/with', exist_ok=False)
os.makedirs('training/without', exist_ok=False)
for framenum in frames_with_fish:
    cv2.imwrite('training/with/%i.png' % framenum, frames[framenum].image)
for framenum in frames_without_fish:
    cv2.imwrite('training/without/%i.png' % framenum, frames[framenum].image)
