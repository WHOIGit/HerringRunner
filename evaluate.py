#!/usr/bin/env python
import argparse
import datetime
import json
import math
import os
import sys

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
with (sys.stdin if args.json == '-' else open(args.json)) as j:
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
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video = None


# The scorecard
true_positives = 0
false_negatives = 0
true_negatives = 0
false_negatives = 0


# Loop over all Fish markers in the database
frames_with_fish = set()
db = clickpoints.DataFile(args.clickpoints)
for marker in db.getMarkers(type=db.getMarkerType('Fish')):
    frames_with_fish.add(marker.image.frame)


# Now figure out which detection ranges contain fish frames
expected_fish = { d: 0 for d in detections }
for frame in frames_with_fish:
    timestamp = datetime.timedelta(seconds=frame / framerate)

    # Search for a detection range containing this frame
    found = False
    for d in detections:
        if d[0] <= timestamp <= d[1]:
            expected_fish[d] += 1
            found = True
            break
        elif d[1] > timestamp:  # exploit sorted order to short circuit
            break

    if not found:
        false_negatives += 1


# Count up all the frames in detected ranges that contained no actual fish
false_positives = 0
for k, v in expected_fish.items():
    if v == 0:
        false_positives += int((k[1] - k[0]).total_seconds() * framerate) + 1


# Count up the frames that were in detected ranges
true_positives = sum(expected_fish.values())


# Count up frames that were *not* in detected ranges, minus the ones that should
# have been
true_negatives = total_frames - false_negatives
for d in detections:
    true_negatives -= int((d[1] - d[0]).total_seconds() * framerate) + 1

assert (true_positives + true_negatives + false_positives + false_negatives) \
    == total_frames


# Calculate the Matthews correlation coefficient score
num = (true_positives * true_negatives - false_positives * false_negatives)
den = (true_positives + false_positives) * \
      (true_positives + false_negatives) * \
      (true_negatives + false_positives) * \
      (true_negatives + false_negatives)
mcc = 0 if den == 0 else (num / math.sqrt(den))


# Summarize
json.dump({
    'frames': total_frames,
    'true_positives': true_positives,
    'false_positives': false_positives,
    'true_negatives': true_negatives,
    'false_negatives': false_negatives,
    'mcc': mcc,
}, sys.stdout, indent=4, sort_keys=True, default=utils.jsonconverter)
print()
