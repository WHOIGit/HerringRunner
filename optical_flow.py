#!/usr/bin/env python3
import argparse
import collections
import math
import multiprocessing
import os
import pickle
import subprocess
import sys
import tempfile

import cv2
import numpy
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str, required=True,
    help='path to output video file')
parser.add_argument('-f', '--flow',
    choices=('farneback', 'tvl1'), default='farneback',
    help='dense optical flow algorithm to use')
parser.add_argument('-n', '--pool-size', type=int, default=None,
    help='number of workers (default = # of cores)')
parser.add_argument('datafile', type=str,
    help='path to a pickled data file')
args = parser.parse_args()


# Data type in the pickled file, from prepare_fast_load.py
Frame = collections.namedtuple('Frame', ('image', 'fish'))
with open(args.datafile, 'rb') as f:
    frames = pickle.load(f)
assert len(frames) > 0
framenums = sorted(frames.keys())


# This function is copied from the opt_flow.py example in OpenCV
def draw_flow_hsv(flow):
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

def draw_flow_intensity(flowmag):
    h, w = flowmag.shape
    hsv = numpy.zeros((h, w, 3), numpy.uint8)
    hsv[...,0] = 0  # hue
    hsv[...,1] = 0  # saturation -- white
    hsv[...,2] = numpy.minimum(flowmag*4, 255)  # value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_optical_flow(framenum):
    #if framenum > 5429: return (framenum, None)
    frame, lastframe = frames[framenum], frames.get(framenum - 1)
    if not lastframe:
        # We cannot compute optical flow because we do not have the frame that
        # preceeds this one
        return (framenum, None)

    # Compute optical flow between the current frame and the last one
    # Resources:
    # * "Learning OpenCV 3" by Kaehler & Bradski, page 590
    if args.flow == 'farneback':
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
            7,  # averaging window size
            3,  # number of iterations at each pyramid level
            5,  # size of pixel neighborhood
            1.2,  # standard deviation used to smooth derivatives
            0  # flags)
        )
    else:
        # * https://stackoverflow.com/a/46602202/145504
        flow = cv2.optflow.createOptFlow_DualTVL1()\
            .calc(lastframe.image, frame.image, None)

    # Compute the flow magnitude
    flowmag = numpy.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    return (framenum, flowmag)


# Loop over each frame and compute its optical flow
print('Computing optical flow...')
with multiprocessing.Pool(processes=args.pool_size) as pool:
    # for multiprocessing progress, https://stackoverflow.com/a/45276885/145504
    flows = dict(tqdm.tqdm(
        pool.imap_unordered(compute_optical_flow, frames.keys()),
        total=len(frames)
    ))

# Forget any frames with no flow information
for framenum in { k for k, v in flows.items() if v is None }:
    framenums.remove(framenum)
    frames.pop(framenum)
    flows.pop(framenum)


# Compute the 90th percentile of flow velocity for the whole video
print('Computing threshold...')
all_flow = numpy.stack(list(flows.values()))
all_flow = all_flow[all_flow >= 1.0]
threshold = numpy.percentile(all_flow, 95)
all_flow = None


# Apply a threshold to the flow magnitudes
print('Applying threshold to flow magnitudes...')
for framenum, flowmag in tqdm.tqdm(flows.items()):
    flowmag[flowmag < threshold] = 0
    flowmag[flowmag >= threshold] = 100.0

# Loop over each computed flow and render out a frame
# XXX Couldn't get cv2.VideoWriter() to work on macOS >:|
framedir = tempfile.TemporaryDirectory()

def draw_frame(i, framenum):
    frame = frames[framenum]
    flowmag = flows[framenum]
    
    # Create a new output frame
    HISTOGRAM_WIDTH = 300
    outframe = numpy.zeros((
        frame.image.shape[0],  # height
        frame.image.shape[1] * 2 + HISTOGRAM_WIDTH,  # width
        3
    ))
    x = 0  # x drawing offset

    # Copy the input frame onto the left side
    outframe[:frame.image.shape[0], x:x+frame.image.shape[1], :] = \
        cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
    x += frame.image.shape[1]

    # Drow the flow visualization
    flowimg = draw_flow_intensity(flowmag)
    outframe[:flowimg.shape[0], x:x+flowimg.shape[1], :] = flowimg
    x += flowimg.shape[1]

    # Draw a border showing whether there are *any* fish
    cv2.rectangle(
        outframe,
        (0, 0), (outframe.shape[1], outframe.shape[0]),  # corners
        (0, 255, 0) if frame.fish else (0, 0, 255),
        2
    )

    # Draw crosshairs on the fish themselves (in both images)
    for marker in frame.fish:
        for off in [(0, 0), (frame.image.shape[1], 0)]:
            cv2.drawMarker(
                outframe,
                (int(marker[0] + off[0]), int(marker[1] + off[1])),  # position
                (0, 255, 0),  # color
                cv2.MARKER_CROSS  # marker type
            )

    # Draw a motion intensity histogram
    diag = int(math.sqrt(frame.image.shape[1]**2 + frame.image.shape[0]**2))
    hist, _ = numpy.histogram(flowmag, bins=200)
    y = 20
    for freq in hist:
        cv2.line(
            outframe,
            (x + 20, y),  # start point
            (int(x + 20 + freq), y),  # end point
            # freq * (HISTOGRAM_WIDTH - 40))
            (255, 255, 255)  # color
        )
        y += 1

    framepath = os.path.join(framedir.name, 'frame_%03i.png' % (i + 1))
    cv2.imwrite(framepath, outframe)

    return numpy.max(flowmag)

# Can't use a lambda expression, for some reason... unpack tuple argument
def wrap_draw_frame(a):
    return draw_frame(*a)

print('Generating output video...')
with multiprocessing.Pool(processes=args.pool_size) as pool:
    list(tqdm.tqdm(
        pool.imap_unordered(wrap_draw_frame, enumerate(framenums)),
        total=len(flows)
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
