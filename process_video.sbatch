#!/usr/bin/env bash

#SBATCH --job-name=process_video
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=slurm/%j.%x.log

#set -eu
#trap date EXIT

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
echo "Command: $0 $@"
cd /vortexfs1/scratch/sbatchelder/herringrunner
hostname; pwd; date


# Error if this is not a GPU node
[ -n "${CUDA_VISIBLE_DEVICES+x}" ]

module load cuda10.1/{toolkit,blas,fft,cudnn}
source herring_yolo_env/bin/activate
echo "Environment... Loaded"

# Get configuration options
VIDEOFILE="$1"
shift
PROCESS_ARGS="$@"

# Get the name of the video without extension
VIDEONAME="$(basename "$VIDEOFILE" | rev | cut -d . -f 2- | rev)"

# make output directory
FRAME_OUT=test-data/video_frames/"$VIDEONAME"
PROCFRAME_OUT=test-data/video_procframes/"$VIDEONAME"
mkdir -p "$FRAME_OUT"
mkdir -p "$PROCFRAME_OUT"


python process_video.py \
    -v "$VIDEOFILE" \
    --progress \
    --ramdisk \
    --num-cores 2 \
    --save-original "$FRAME_OUT" \
    --save-preprocessed "$PROCFRAME_OUT" \
    $PROCESS_ARGS
    

