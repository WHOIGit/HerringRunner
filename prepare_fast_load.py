#!/usr/bin/env python3
import argparse
import collections
import pickle

import clickpoints
import cv2
import intervaltree
import numpy
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True,
    help='path to the video file')
parser.add_argument('-c', '--clickpoints', required=True,
    help='path to the clickpoints file')
parser.add_argument('-o', '--output', required=True,
    help='path to write out the pickled object')
parser.add_argument('--before', default=3,
    help='number of seconds\' worth of frames to extract before a fish')
parser.add_argument('--after', default=3,
    help='number of seconds\' worth of frames to extract after a fish')
args = parser.parse_args()


# Load the video
video = cv2.VideoCapture(args.video)
framerate = video.get(cv2.CAP_PROP_FPS)
nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Identify frame numbers with fish
frames_with_fish = set()
cdb = clickpoints.DataFile(args.clickpoints)
for marker in cdb.getMarkers(type=cdb.getMarkerType('Fish')):
    frames_with_fish.add(int(marker.image.frame))

# Create a tree of intervals before and after each fish frame
# Note: Intervals do not include the upper bound 
tree = intervaltree.IntervalTree()
before = after = 3  # seconds
for frame in frames_with_fish:
    tree[max(0, int(frame - args.before * framerate)):\
         min(int(frame + args.after * framerate) + 1, nframes)] = None

# Simplify the tree to merge overlapping intervals
tree.merge_overlaps()

# Extract the frames
Frame = collections.namedtuple('Frame', ('image', 'fish'))
frames = {}
for framenum in tqdm.tqdm(range(tree.end())):
    success, frame = video.read()
    if not success:
        continue

    # Extract the frame if it's in the interval tree
    if tree.overlaps(framenum):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[framenum] = Frame(image=gray, fish=[])

# Add fish markers to frame metadata
for marker in cdb.getMarkers(type=cdb.getMarkerType('Fish')):
    frames[marker.image.frame].fish.append((marker.x, marker.y))

# Save the data to our output file
with open(args.output, 'wb') as f:
    pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
