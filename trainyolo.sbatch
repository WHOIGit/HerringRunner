#!/usr/bin/env bash

#SBATCH --job-name=trainyolo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/%j.%x.log


echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
echo "Command: $0 $@"
hostname; pwd; date; echo "CUDA=$CUDA_VISIBLE_DEVICES"

module load cuda10.1/{toolkit,blas,fft,cudnn}
source herring_yolo_env/bin/activate
echo "Environment... Loaded"

set -x

DATAYML=$1
MODELNAME=$2

# move into yoloV5 directory
cd yolov5_ultralytics 

echo; echo "TRAINING START"
time python train.py --data "../$DATAYML" --name "$MODELNAME" --project "../training-output" --weights yolov5m.pt --adam --device $CUDA_VISIBLE_DEVICES --batch-size 32 --epochs 800


# run TEST labels if provided

if [ "$#" -eq 3 ]; then
    TESTDATA=$3
    
    echo; echo "TESTSET DETECT"    
    time python detect.py --source "../$TESTDATA" --weights "../training-output/$MODELNAME/weights/best.pt" --project "../training-output/$MODELNAME" --name "testset_detect" --save-txt --nosave --hide-labels --device $CUDA_VISIBLE_DEVICES 
    
    
    echo; echo "TESTSET SUMMARY"
    python ../detect_summary.py "../training-output/$MODELNAME/testset_detect/labels" --input-labels "../$TESTDATA" -o "../training-output/$MODELNAME/testset_detect/stats" --model-name $MODELNAME
    
fi
   

echo; echo Job ID: $SLURM_JOB_ID is DONE!

