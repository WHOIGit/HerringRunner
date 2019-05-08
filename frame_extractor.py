#!/usr/bin/env python
'''
This script takes a video and splits it into frames. Requires ffmpeg(1).
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
    help='directory in which to output frames')
  parser.add_argument('-i', '--interval', required=True, type=float,
    help='frame interval (seconds)')
  parser.add_argument('-j', '--json', help='JSON file containing detections')
  parser.add_argument('-s', '--start',
    help='extract frames from this timestamp')
  parser.add_argument('-e', '--end',
    help='extract frames until this timestamp')
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

  # Build up the FFmpeg commands to execute
  cmds = []
  for i, (start, end) in enumerate(detections):
    cmd = ['ffmpeg', '-hide_banner']
    cmd.extend(['-i', args.video])  # input file
    if start:
      cmd.extend(['-ss', str(start)])  # start time
    if end:
      # In case start == end, bump end by a fraction of a second so that ffmpeg
      # does not complain
      end += datetime.timedelta(milliseconds=1)
      cmd.extend(['-to', str(end)])  # stop time
    cmd.extend(['-vf', 'fps=1/%f' % args.interval])  # fps video filter
    cmd.append(os.path.join(args.out, 'frame_%i_%%06d.png' % i))
    cmds.append(cmd)

  # Dry run: just print the commands
  if args.dry_run:
    for cmd in cmds:
      print(' '.join(shlex.quote(x) for x in cmd))
    sys.exit(0)

  # For safety, do not output to a non-empty directory
  if not os.path.exists(args.out):
    os.makedirs(args.out)
  elif os.listdir(args.out):
    print('Error: %s: Directory not empty' % args.out, file=sys.stderr)
    sys.exit(1)

  # Record the arguments we were called with
  with open(os.path.join(args.out, 'info.json'), 'w') as f:
    json.dump({
      'args': vars(args),
      'cmds': cmds
    }, f, indent=4, sort_keys=True, default=utils.jsonconverter)
    f.write('\n')

  # Invoke the command
  for cmd in cmds:
    subprocess.check_call(cmd)
