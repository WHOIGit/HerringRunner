# HerringRunner

Scripts for detecting herring runs in a fish ladder at the Cape Cod Canal. Separate scripts for detection and extraction of images for [Zooniverse][].

[Zooniverse]: https://zooniverse.org

* `detector.py`: Uses a modified MOG background subtraction algorithm to detect motion in a video clip of the fish ladder. Outputs time ranges within the video that contain detected objects.

* `clip_extractor.py`: Takes in a video and extracts a clip from it.

* `frame_extractor.py`: Takes in a video and splits it up into frames at the specified interval.

* `was_detected.py`: Takes in a timestamp and a JSON file and decides if anything was detected at that timestamp or not.


## Setup

    sudo apt-get install ffmpeg python3 python3-venv
    git clone https://github.com/WHOIGit/HerringRunner.git
    cd HerringRunner
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
