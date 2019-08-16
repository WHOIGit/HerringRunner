#!/usr/bin/env python3
import argparse
import collections
import pickle

import cv2
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str, required=True,
    help='path to output pickled data file')
parser.add_argument('datafile', type=str,
    help='path to a pickled data file')
args = parser.parse_args()


# Data type in the pickled file, from prepare_fast_load.py
Frame = collections.namedtuple('Frame', ('image', 'fish'))
with open(args.datafile, 'rb') as f:
    frames = pickle.load(f)
assert len(frames) > 0

# Loop over frames in order, subtracting the background
bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
gradient = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

framenums = sorted(frames.keys())
outframes = {}
for framenum in tqdm.tqdm(framenums):
    frame = frames[framenum]
    
    # Blur the frame to eliminate small bits of noise
    blurred = cv2.GaussianBlur(frame.image, (21, 21), 0)
    mask = bg.apply(blurred)

    # Performing a morphological opening to eliminate small dots in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, gradient)

    image = cv2.bitwise_and(frame.image, frame.image, mask=mask)
    outframes[framenum] = Frame(image=image, fish=frame.fish)
    #frames[framenum]._replace(image=image)  # not sure why it doesn't work

# Delete the first frame, which will just be blank
del outframes[framenums[0]]

# Save the data to our output file
with open(args.output, 'wb') as f:
    pickle.dump(outframes, f, protocol=pickle.HIGHEST_PROTOCOL)
