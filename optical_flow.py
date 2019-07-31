#!/usr/bin/env python3
import argparse
import collections
import multiprocessing
import os
import pickle
import subprocess
import tempfile

import cv2
import numpy
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str, required=True,
    help='path to output video file')
parser.add_argument('-n', '--pool-size', type=int, default=None,
    help='number of workers (default = # of cores)')
parser.add_argument('--subtract-bg', action='store_true',
    help='subtract the background before calculating optical flow')
parser.add_argument('datafile', type=str,
    help='path to a pickled data file')
args = parser.parse_args()


# Data type in the pickled file, from prepare_fast_load.py
Frame = collections.namedtuple('Frame', ('image', 'fish'))
with open(args.datafile, 'rb') as f:
    frames = pickle.load(f)
assert len(frames) > 0


# This function is copied from the opt_flow.py example in OpenCV
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = numpy.arctan2(fy, fx) + numpy.pi
    v = numpy.sqrt(fx*fx+fy*fy)
    hsv = numpy.zeros((h, w, 3), numpy.uint8)
    hsv[...,0] = ang*(180/numpy.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = numpy.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def process_frame(i, framenum):
    frame, lastframe = frames[framenum], frames.get(framenum - 1)
    if not lastframe:
        # We cannot compute optical flow because we do not have the frame that
        # preceeds this one
        return

    # Resources:
    # * "Learning OpenCV 3" by Kaehler & Bradski, page 590
    # * Python example:
    #     https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    # * Outdated Python docs:
    #     https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#cv2.calcOpticalFlowFarneback
    flow = cv2.calcOpticalFlowFarneback(
        lastframe.image,
        frame.image,
        None,  # initial computed flow
        0.5,  # image pyramid scale
        3,  # number of pyramid levels
        15,  # averaging window size
        3,  # number of iterations at each pyramid level
        5,  # size of pixel neighborhood
        1.2,  # standard deviation used to smooth derivatives
        0  # flags)
    )

    #image = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
    image = draw_hsv(flow)

    # Draw a border showing whether there are *any* fish
    cv2.rectangle(
        image,
        (0, 0), (image.shape[1], image.shape[0]),  # corners
        (0, 255, 0) if frame.fish else (0, 0, 255),
        2
    )

    # Draw crosshairs on the fish themselves
    for marker in frame.fish:
        cv2.drawMarker(
            image,
            (int(marker[0]), int(marker[1])),  # position
            (0, 255, 0),  # color
            cv2.MARKER_CROSS  # marker type
        )

    framepath = os.path.join(framedir.name, 'frame_%03i.png' % (i + 1))
    cv2.imwrite(framepath, image)


# Subtract the background from each frame, if desired
framenums = sorted(frames.keys())
if args.subtract_bg:
    bg = cv2.createBackgroundSubtractorMOG2()
    for framenum in tqdm.tqdm(framenums):
        image = frames[framenum].image
        mask = bg.apply(image)
        image = cv2.bitwise_and(image, image, mask=mask)
        frames[framenum] = frames[framenum]._replace(image=image)
    
    # Delete the first frame, because it's going to be blank
    del frames[framenums[0]]
    del framenums[0]

    # Release the background subtractor to free memory
    bg = None


# Loop over each frame and write it out
# XXX Couldn't get cv2.VideoWriter() to work on macOS >:|
framedir = tempfile.TemporaryDirectory()

# This lets us use pool.imap*(); can't use a lambda expression??
def apply_process_frame(a):
    process_frame(*a)

with multiprocessing.Pool(processes=args.pool_size) as pool:
    list(tqdm.tqdm(  # for progress, https://stackoverflow.com/a/45276885/145504
        pool.imap_unordered(apply_process_frame, enumerate(framenums)),
        total=len(frames)
    ))

# Invoke ffmpeg to convert the frames to a video
try:
    subprocess.check_call([
        'ffmpeg',
        '-framerate', str(5),
        '-i', os.path.join(framedir.name, 'frame_%03d.png'),  # %d not %i!
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        args.output,
    ])
except subprocess.CalledProcessError:
    print('FFmpeg failed! Check', framedir.name)
    while True:
        pass
framedir = None  # deletes the frame files
