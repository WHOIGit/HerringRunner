#!/usr/bin/env python
import argparse
import datetime
import json
import os

import clickpoints
import cv2

import utils


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True,
    help='path to the video file')
parser.add_argument('-c', '--clickpoints', required=True,
    help='path to ClickPoints database')
parser.add_argument('-j', '--json', help='JSON file containing detections')
args = parser.parse_args()


# Load detection ranges from the JSON file
detections = []
with open(args.json) as j:
    data = json.load(j)
    assert os.path.basename(data['video']) == os.path.basename(args.video)
    for start, end in data['detections']:
        detections.append((
            utils.parse_duration(start),
            utils.parse_duration(end)
        ))


# Get framerate information from the video
video = cv2.VideoCapture(args.video)
framerate = video.get(cv2.CAP_PROP_FPS)
video = None


# A map of detection ranges to actual fish
detection_fish = { d: 0 for d in detections }

# A count of fish that were not in detected frames
undetected_fish = 0


# Loop over all Fish markers in the database
db = clickpoints.DataFile(args.clickpoints)
for marker in db.getMarkers(type=db.getMarkerType('Fish')):
    timestamp = datetime.timedelta(seconds=marker.image.frame / framerate)

    # Search for a detection range containing this fish
    found = False
    for d in detections:
        if d[0] <= timestamp <= d[1]:  # inclusive OK?
            detection_fish[d] += 1
            found = True
            break
        elif d[1] > timestamp:  # exploit sorted order to short circuit
            break

    if not found:
        undetected_fish += 1

# Count the number of detection ranges that didn't actually contain any fish
false_positives = 0
for v in detection_fish.values():
    if v == 0:
        false_positives += 1

# Summarize
print('True positives:', len(detections) - false_positives, 'detection ranges')
print('False positives:', false_positives, 'detection ranges')
print('False negatives:', undetected_fish, 'missed fish')
