#!/usr/bin/env python
'''
This script determines whether anything was detected at a specific timestamp.
'''
import argparse
import datetime
import json
import sys

import utils


if __name__ == '__main__':
  # Parse arguments
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-t', '--timestamp', required=True,
    help='timestamp of an expected detection')
  parser.add_argument('-j', '--json', required=True,
    help='JSON file containing detections')
  args = parser.parse_args()

  args.timestamp = utils.parse_duration(args.timestamp)

  detections = []
  with open(args.json) as j:
    data = json.load(j)
    for start, end in data['detections']:
      detections.append((
        utils.parse_duration(start),
        utils.parse_duration(end)
      ))
  
  for start, end in detections:
    if start <= args.timestamp <= end:
      print('Detected between', start, 'and', end)
      sys.exit(0)
  
  print('Not detected')
  sys.exit(1)
