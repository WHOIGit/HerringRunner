#!/usr/bin/env python
import argparse
import datetime
import os
import time

import cv2
import imutils

from pyimagesearch.keyclipwriter import KeyClipWriter


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    '-o', '--output', required=True, help='path to output directory')
ap.add_argument('-v', '--video', required=True, help='path to the video file')
ap.add_argument(
    '-f', '--fps', type=int, default=20, help='FPS of output video')
ap.add_argument(
    '-c', '--codec', type=str, default='MJPG', help='codec of output video')
ap.add_argument(
    '-b',
    '--buffer-size',
    type=int,
    default=35,
    help='buffer size of video clip writer')
args = ap.parse_args()

# initialize the video stream
vs = cv2.VideoCapture(args.video)
vidname = os.path.basename(args.video)

# initialize key clip writer and the consecutive number of
# frames that have *not* contained any action
kcw = KeyClipWriter(bufSize=args.buffer_size)
consecFrames = 0
#frame count is for timer only, must be done for when cv2.CAP_PROP_POS_MSEC
# is empty, like in most avi files
frame_count = 0
# initialize the first frame in the video stream
avgFrame = None

#frame count is for timer only, must be done for when cv2.CAP_PROP_POS_MSEC
# is empty, like in most avi files
frame_count = 0

debug_var = False

# make the output folder
if not os.path.exists(args.output):
    os.mkdir(args.output)
else:
    print('Warning: output directory exists')

# keep looping
while True:
    # grab the current frame, resize it, and initialize a
    # boolean used to indicate if the consecutive frames
    # counter should be updated
    frame = vs.read()
    frame = frame if args.video is None else frame[1]
    #frame = imutils.resize(frame, width=600)
    f_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
    surface = f_width * f_height
    currentsurface = 0
    updateConsecFrames = True
    #track
    trackframe = vs.get(cv2.CAP_PROP_POS_FRAMES)
    #get timestamp
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_count += 1
    time = int(float(frame_count) / fps)
    timestamp = datetime.datetime.utcfromtimestamp(time)
    #print(timestamp)

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

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

    #DEBUG count countours
    #print (len(cnts))
    #print (str(timestamp.strftime('%H:%M:%S')))
    for c in contours:
        # compute the area for each contour and increment surface
        currentsurface += cv2.contourArea(c)
    #backcontours = contours #Save contours
    avgsurf = (
        currentsurface * 100
    ) / surface  #Calculate the average of contour area on the total size

    # only proceed if at least one contour was found
    if avgsurf >= 21:
        count = avgsurf
        print(str(count) + ' / ' + str(timestamp.strftime('%H:%M:%S')))
        updateConsecFrames = False
        consecFrames = 0
        # if we are not already recording, start recording
        if not kcw.recording:
            #get timestamp
            #times=vs.get(cv2.CAP_PROP_POS_MSEC)
            #timestamp = datetime.datetime.utcfromtimestamp(times//1000.0)
            #use trac
            #try to reverse past to one sec before clip, approx 30 frames
            vs.set(1, frame_count - args.buffer_size)

            debug_var = True
            p = '{}/{}.avi'.format(
                args.output,
                str(vidname) + '_' + str(timestamp.strftime('%H:%M:%S')))
            kcw.start(p, cv2.VideoWriter_fourcc(*args.codec), 10)

    # otherwise, no action has taken place in this frame, so
    # increment the number of consecutive frames that contain
    # no action
    if updateConsecFrames:
        consecFrames += 1

    # update the key frame clip buffer
    kcw.update(frame)
    # if we are recording and reached a threshold on consecutive
    # number of frames with no action, stop recording the clip
    if kcw.recording and consecFrames == (args.buffer_size + 2):
        print('videoclip ' + p)
        kcw.finish()

    # show the frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break

# if we are in the middle of recording a clip, wrap it up
if kcw.recording:
    kcw.finish()

# do a bit of cleanup

if debug_var == True & args.debugger == False:
    print('videos recorded, check output')
else:
    print('no videos recorded!')

vs.stop() if args.video is None else vs.release()
cv2.destroyAllWindows()
