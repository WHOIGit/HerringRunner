#!/usr/bin/env python
'''
This script takes a video and extracts a clip. Requires ffmpeg(1).
'''
import argparse
import json
import os
import shlex
import subprocess
import sys


if __name__ == '__main__':
  # Parse arguments
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-v', '--video', required=True,
    help='path to the video file')
  parser.add_argument('-o', '--out', required=True,
    help='output video file')
  parser.add_argument('-s', '--start', required=True,
    help='extract frames from this timestamp')
  parser.add_argument('-e', '--end', required=True,
    help='extract frames until this timestamp')
  parser.add_argument('--dry-run', action='store_true',
    help='just display command to run, do not run it')
  args = parser.parse_args()

  # Build up the FFmpeg command
  cmd = ['ffmpeg', '-hide_banner']
  cmd.extend(['-i', args.video])  # input file
  cmd.extend(['-c', 'copy'])
  if args.start:
    cmd.extend(['-ss', args.start])  # start time
  if args.end:
    cmd.extend(['-to', args.end])  # stop time
  cmd.append(args.out)

  # Dry run: just print the command
  if args.dry_run:
    print(' '.join(shlex.quote(x) for x in cmd))
    sys.exit(0)

  # Invoke the command
  subprocess.check_call(cmd)
