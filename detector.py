#!/usr/bin/env python
'''
This script detects herring in a video file and outputs timetsamps at which the
herring were found.
'''
import cv2

import utils


def preprocess(frame, blur_factor, blur_type='gaussian'):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_factor > 0:
        if blur_type == 'gaussian':
            gray = cv2.GaussianBlur(gray, (blur_factor, blur_factor), 0)
        elif blur_type == 'median':
            gray = cv2.medianBlur(gray, blur_factor)
        else:
            raise NotImplementedError()
    # gray = cv2.createCLAHE().apply(gray)  # very strange results
    return gray


def subtract_background(bg, frame, bg_threshold):
    if bg is None:
        bg = cv2.createBackgroundSubtractorMOG2(varThreshold=bg_threshold, detectShadows=False)
    mask = bg.apply(frame)
    return bg, cv2.bitwise_and(frame, frame, mask=mask)


def process(frame, threshold=0, dilations=0):
    # Turn any pixel that is brighter than our threshold white
    if threshold > 0:
        _, thresh = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    else:
        thresh = frame

    # Dilate the thresholded image to fill in holes
    if dilations > 0:
        thresh = cv2.dilate(thresh, None, iterations=dilations)

    return thresh


def measure_coverage(frame, vid_area):
    # Find contours within the image.
    # (Note, this function returns a different number of parameters on different
    # versions of OpenCV.)
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the fraction of the image covered by the contours
    coverage = sum(cv2.contourArea(c) for c in contours) / vid_area
    return coverage


if __name__ == '__main__':
    import argparse
    import datetime
    import json
    import os
    import sys
    import time


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
    group.add_argument('--blur-type', type=str, choices=('median', 'gaussian'),
        default='gaussian', help='blur algorithm to use')
    group.add_argument('--blur-factor', type=int, default=25,
        help='size of the blur kernel (must be odd) (0 = off)')
    group.add_argument('--threshold', type=int, default=5,
        help='brightness threshold (0 = off)')
    group.add_argument('--dilations', type=int, default=2,
        help='number of dilation iterations (0 = off)')
    group.add_argument('--bg-threshold', type=float, default=16.0,
        help='background subtractor threshold')
    args = parser.parse_args()

    # Validate arguments
    assert args.blur_factor == 0 or args.blur_factor % 2 == 1
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

        # Preprocess
        frame = preprocess(frame, blur_factor=args.blur_factor,
            blur_type=args.blur_type)

        # Update the background
        bg, frame = subtract_background(bg, frame,bg_threshold=args.bg_threshold)

        # Process
        processed = process(frame, threshold=args.threshold,
            dilations=args.dilations)

        # Calculate coverage
        coverage = measure_coverage(processed, vid_area)

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
        'settings': vars(args),
        'detections': detections,
    }, sys.stdout, indent=4, sort_keys=True, default=utils.jsonconverter)
    print()
