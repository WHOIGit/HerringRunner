# HerringRunner
Scripts for detecting herring runs in a fishladder at the cape-cod canal. Seperate scripts for detection and extraction of images for zooniverse

* `detection_recorder.py`: Uses a modified MOG background subtraction algorithum to detect motion in a videoclip of the fish ladder. Accepts a path to source video and outputs to a directory any clips of video in which motion is detected.

* `image_extraction.py`: Take a directory of videoclips and extracts a number of still image frames based on input transversal factor (in seconds). Each videoclip has their own directrory with extracted images. 

* `mooring.yml`: Script with a convient CLI for running both detection and image extraction.
