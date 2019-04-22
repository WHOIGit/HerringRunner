#!/usr/bin/env python
import argparse
import datetime
import os
import time

import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='path to the video file')
ap.add_argument(
    '-b',
    '--buffer-size',
    type=int,
    default=35,
    help='buffer size of video clip writer')
args = ap.parse_args()

# initialize the video stream
vs = cv2.VideoCapture(args.video)

# initialize key clip writer and the consecutive number of
# frames that have *not* contained any action
consecFrames = 0
# initialize the first frame in the video stream
avgFrame = None


# Compute the size of a frame
vid_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
vid_area = vid_width * vid_height

# Get the video's framerate
vid_fps = vs.get(cv2.CAP_PROP_FPS)


frame_count = 0
recording = False
while True:
    # Grab the next frame
    success, frame = vs.read()
    if not success:
        break
    
    updateConsecFrames = True
    
    frame_count += 1
    
    # TODO: Read CAP_PROP_POS_MSEC before trying to compute this ourselves
    # TODO: Should that be done _before_ we read the frame?
    
    time = int(float(frame_count) / vid_fps)
    timestamp = datetime.datetime.utcfromtimestamp(time)

    #  convert frame to grayscale, and blur with 'blursize'
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    # if the average frame is None, initialize it
    if avgFrame is None:
        print('[INFO] starting background model...')
        avgFrame = gray.copy().astype('float')
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avgFrame, 0.6)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avgFrame))
    _, thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)

    # NOTE: This function returns a different number of parameters on different
    # versions of OpenCV.
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the fraction of the image covered by contours
    coverage = sum(cv2.contourArea(c) for c in contours) / vid_area

    # only proceed if at least one contour was found
    if coverage >= 0.21:
        print('%f / %s' % (coverage, timestamp.strftime('%H:%M:%S')))
        updateConsecFrames = False
        consecFrames = 0
        if not recording:
            print('Starting recording on frame', frame_count)
            # TODO: We will want to rewind the clip a fraction of a second
            recording = True

    # otherwise, no action has taken place in this frame, so
    # increment the number of consecutive frames that contain
    # no action
    if updateConsecFrames:
        consecFrames += 1

    # if we are recording and reached a threshold on consecutive
    # number of frames with no action, stop recording the clip
    if recording and consecFrames == (args.buffer_size + 2):
        print('Stopping recording on frame', frame_count)
        recording = False
