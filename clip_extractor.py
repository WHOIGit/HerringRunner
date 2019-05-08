#!/usr/bin/env python
'''
This script takes a video and extracts a clip. Requires ffmpeg(1).
'''
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys

import utils


if __name__ == '__main__':
  # Parse arguments
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-v', '--video', required=True,
    help='path to the video file')
  parser.add_argument('-o', '--out', required=True,
    help='output video file')
  parser.add_argument('-j', '--json', help='JSON file containing detections')
  parser.add_argument('-s', '--start',
    help='extract clip from this timestamp')
  parser.add_argument('-e', '--end',
    help='extract clip until this timestamp')
  parser.add_argument('--dry-run', action='store_true',
    help='just display command to run, do not run it')
  args = parser.parse_args()

  detections = []

  if args.json:
    with open(args.json) as j:
      data = json.load(j)
      assert os.path.basename(data['video']) == os.path.basename(args.video)
      for start, end in data['detections']:
        detections.append((
          utils.parse_duration(start),
          utils.parse_duration(end)
        ))

  if args.start and args.end:
    start = utils.parse_duration(args.start)
    end = utils.parse_duration(args.end)
    detections.append((start, end))

  if not detections:
    # Dummy detection that just says extract every frame
    detections.append((None, None))

  # Build up the FFmpeg command
  cmds = []
  for i, (start, end) in enumerate(detections):
    cmd = ['ffmpeg', '-hide_banner']
    cmd.extend(['-i', args.video])  # input file
    cmd.extend(['-c', 'copy'])
    if start:
      cmd.extend(['-ss', str(start)])  # start time
    if end:
      # In case start == end, bump end by a fraction of a second so that ffmpeg
      # does not complain
      end += datetime.timedelta(milliseconds=1)
      cmd.extend(['-to', str(end)])  # stop time
    
    if len(detections) == 1:
      cmd.append(args.out)
    else:
      filename, ext = os.path.splitext(args.out)
      cmd.append('%s_%i%s' % (filename, i, ext))
    cmds.append(cmd)

  # Dry run: just print the command
  if args.dry_run:
    for cmd in cmds:
     print(' '.join(shlex.quote(x) for x in cmd))
    sys.exit(0)

  # Invoke the command
  for cmd in cmds:
    subprocess.check_call(cmd)
