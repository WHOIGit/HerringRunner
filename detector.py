#!/usr/bin/env python
'''
This script detects herring in a video file and outputs timetsamps at which the
herring were found.
'''
import argparse
import datetime
import json
import os
import sys
import time

import cv2

import utils


# Parse arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-v', '--video', required=True,
    help='path to the video file')

parser.add_argument('--interesting', type=float, default=0.21,
    help='fraction of video that must be covered by detected objects to be ' \
         'considered of interest (between 0.0 and 1.0)')
parser.add_argument('--timeout', type=float, default=2,
    help='number of seconds to wait for an interesting object before ' \
         'splitting the clip') 

group = parser.add_argument_group('image processing options')
group.add_argument('--accelerate', action='store_true',
    help='enable GPU acceleration (experimental)')
group.add_argument('--blur-factor', type=int, default=25,
    help='size of the Gaussian blur kernel (must be odd) (0 = off)')
group.add_argument('--threshold', type=int, default=5,
    help='brightness threshold (0 = off)')
group.add_argument('--dilations', type=int, default=2,
    help='number of dilation iterations (0 = off)')
group.add_argument('--bg-weight', type=float, default=0.6,
    help='background average weight for current frame (between 0.0 and 1.0)')
args = parser.parse_args()


# Validate arguments
assert args.blur_factor % 2 == 1
assert 0.0 <= args.bg_weight <= 1.0
args.timeout = datetime.timedelta(seconds=args.timeout)


# This will hold the background image, which will be subtracted off
bg = None

# Initialize the video stream
vs = cv2.VideoCapture(args.video)

# Compute the size of a frame
vid_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
vid_area = vid_width * vid_height

# Get the video's framerate
vid_fps = vs.get(cv2.CAP_PROP_FPS)

# This is the timestamp we started recording on, or None is we aren't recording
first_interesting = None

# This is the timestamp of the last interesting frame
last_interesting = None

# This is the frame number we are on
frame_count = -1

detections = []

while True:
    # Read the timestamp of the next frame (in milliseconds)
    # TODO: This may not be supported for some video formats; if it is not, we
    # will need to figure it out ourselves from the frame number
    timestamp = datetime.timedelta(milliseconds=vs.get(cv2.CAP_PROP_POS_MSEC))

    # Grab the next frame
    success, frame = vs.read()
    if not success:
        break
    frame_count += 1
    
    # Secret sauce: Convert the frame to a unified matrix, so we can use GPU
    # acceleration
    if args.accelerate:
        frame = cv2.UMat(frame)

    # Convert frame to grayscale and apply a blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if args.blur_factor > 0:
        gray = cv2.GaussianBlur(gray, (args.blur_factor, args.blur_factor), 0)

    # If we do not yet have a background, initialize it to this one
    if bg is None:
        bg = cv2.UMat(gray) if args.accelerate else gray.copy().astype('float')
        continue
    
    # Update the background: accumulate this blurred frame, weighted more
    # heavily than all past frames
    if args.accelerate:
        # accumulateWeighted does not seem to be available?
        bg = cv2.addWeighted(bg, 1.0 - args.bg_weight, gray, args.bg_weight, 0.0)
    else:
        cv2.accumulateWeighted(gray, bg, args.bg_weight) 

    # Remove the background from the image
    delta = cv2.absdiff(gray, cv2.convertScaleAbs(bg))
    
    # Turn any pixel that is brighter than our threshold white
    if args.threshold > 0:
        _, thresh = cv2.threshold(delta, args.threshold, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in holes
    if args.dilations > 0:
        thresh = cv2.dilate(thresh, None, iterations=args.dilations)

    # Find contours within the image.
    # (Note, this function returns a different number of parameters on different
    # versions of OpenCV.)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the fraction of the image covered by the contours
    coverage = sum(cv2.contourArea(c) for c in contours) / vid_area

    # If enough of the image is interesting, start recording
    if coverage >= args.interesting:
        last_interesting = timestamp
        first_interesting = first_interesting or timestamp

    # When we have enough uninteresting frames in a row, stop
    if first_interesting is not None:
        if (timestamp - last_interesting) >= args.timeout:
            detections.append((first_interesting, last_interesting))
            first_interesting = None

if first_interesting is not None:
    # We need to end the current string of interesting frames
    detections.append((first_interesting, timestamp))

json.dump({
    'video': os.path.basename(args.video),
    'detections': detections
}, sys.stdout, indent=4, sort_keys=True, default=utils.jsonconverter)
print()