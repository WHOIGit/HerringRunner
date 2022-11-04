# Herring Runner

This project allows a user to train and run models for the purpose of counting fish from a fixed camera on a fish ladder.
It leverages [ultralytics' yolov5](https://github.com/ultralytics/yolov5), [rgov's FishDetect](https://github.com/WHOIGit/FishDetector) for its implementation of Salman et. al.'s "Automatic fish detection in underwater videos by a deep neural network-based hybrid motion learning system" algorith, and optionally the [IFCB Classifier](https://github.com/WHOIGit/ifcb_classifier) as an alternate approach to counting fish in frames. 


## Installation

```
sudo apt-get install ffmpeg python3 python3-venv
git clone https://github.com/WHOIGit/HerringRunner.git
cd HerringRunner
python -m venv herring_yolo_env
source herring_yolo_env/bin/activate
pip install -r yolov5_ultralytics/requirements.txt
```
Then install opencv2 with cuda on the hpc by following [these steps](https://github.com/WHOIGit/FishDetector#building-opencv).


To optionally try using a modified classifier model (instead of yolo) to count fish, additionally create this secondary `herring_classnn_env` environment.

```
source deactivate
python -m venv herring_classnn_env
source herring_classnn_env/bin/activate
pip install -r pytorch_classifier/requirements/requirements.txt
``` 


## Script Usage Overview

`generate_training_lists.py` - recursively access .txt label files in a given directory, outputs lists of frame image files according to various config parameters. The ratio of training, validation, and test frames, as well as the ratio of null frames (frames not-containing fish) can be adjusted. The list files can then be referenced in a yolo config.yml for training, or the classifier model trainer. 


`trainclassnn.sbatch` - uses herring_classnn_env environment. Accepts (1) a dataset configuration directory (eg training-data/lists/EXAMPLE_DIR) containing "training.txt" and "validation.txt"; and (2) a base model classifier architecture (eg inception_v3 or resnet101); optionally (3) a test-set of frame files (and associated label files) from which training statistics are derived. 


`trainyolo.sbatch` - Accepts (1) yaml file defining the training data, (2) the model name for output, (3) optionally, a test-set of frame files (and associated label files) from which training statistics are derived. 


`ffprob_videos.batch` - a bash script that accepts (1) a list file of fullpath video files, and (2) an output filename csv. This bash script collates video duration and total frame counts for a list of videos and outputs the results as a csv. 


`process_video.py` - This script processes a video file into raw frames and processed video frames. Processed video frames are bassed on the algorithm described in "Automatic fish detection in underwater videos by a deep neural network-based hybrid motion learning system" by Salman, et al. (2019). OpenCV2 must be properly installed for this script to create processed frames. This script has a large number of configurable parameters, to view them all you may use the `--help` flag to display them. 


`process_video.sbatch` - This script converts a video file into frames and processed frames. It accepts (1) a video file path, and therafter optionally arguments for `process_video.py`. Some of the python file arguments are automatically included and these are "--progress --ramdisk --num-cores 2 --save-original --save-preprocessed". The --save-original --save-preprocessed arguments are automatically set to "test-data/video_frames/VIDEONAME" and "test-data/video_procframes/VIDEONAME" respectively. 


`detect.sbatch` - This script applies a trained yolo model to a list of frame image. It accepts (1) a yolo .pt model, (2) a listfile of frame image paths or a directory path containing frame images, (3) an output directory NAME. The results get output to "detect-output/NAME".


`process_video_list.sbatch` - This script processes and optionally runs inference on a given list of videos. It leverages slurm's array capabilities such that each array index corresponds to a particular video in a list; as such the --array slurm argument must be supplied for this sbatch script to work. This scripts accept (1) a listfile of video fullpaths. If no further arguments are supplied, this script outputs raw and processed frames to `detect-data/video_rawframes/VIDEONAME` and `detect-data/video_procframes/VIDEONAME` where VIDEONAME is the name of a particular array-given video. Additionally, two textlists of extracted video raw and processed frames are created under `detect-data/video_framelist`. Optionally, (2) a model file may be specified. The model may be a yolo `best.pt` file or a pytorch classifier model. If a trained classifier model is used, the herring_yolo_env environment is deactivated and the herring_classnn_env environment is activated. When a model is specified, this script outputs model results and cleans up raw and processed frame files from disk after the output model results are calculated. Model results per video get saved under `detect-output/MODELNAME/VIDEONAME.csv`. It must be noted that this script assumes that the `afr.json` Adaptive Frame Rate file is present in the project directory; this addition means that not all video frames will be processed. 


`afr.json` - Adaptive Frame Rate json file. This file specifies how many frames to skip during fish detection for a given month. This data is used to account for different mean fish swim speeds so as to avoid double-counting fish.


`agg_multi.py` - Adds weather metadata to detection results csv. Accepts (1) an input csv with the following columns ""; and (2) an output csv filename. This script references "weather-daily_2017-06-01_2019-12-31.csv" and "weather-hourly_2017-06-01_2019-12-31.csv" for its data. Weather data is from visualcrossing.com


