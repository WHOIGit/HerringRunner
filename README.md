# HerringRunner

Scripts for detecting herring runs in a fish ladder at the Cape Cod Canal. Separate scripts for detection and extraction of images for [Zooniverse][].

[Zooniverse]: https://zooniverse.org

* `detection_recorder.py`: Uses a modified MOG background subtraction algorithm to detect motion in a video clip of the fish ladder. Accepts a path to source video and outputs to a directory any clips of video in which motion is detected.

* `image_extractor.py`: Takes in a video and splits it up into frames at the specified interval (in milliseconds).

* `runner.py`: Script with a convient CLI for running both detection and image extraction.

## Setup

    sudo apt-get install ffmpeg python3 python3-venv
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

## Timing

Time spacing based on the median number of frames it took a fish to pass by. Since the footage was 15 fps, those times by month are:

* June: 6/15 = 0.4 s
* July: 5/15 = 0.33 s
* August: 3/15 = 0.2 s
* Sept: 4/15 = 0.27 s
* Oct: 4/15 = 0.27 s
* Nov: 13/15 = 0.87 s


