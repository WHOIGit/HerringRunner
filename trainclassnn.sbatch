#!/usr/bin/env bash

#SBATCH --job-name=trainclassnn
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
source herring_classnn_env/bin/activate
echo "Environment... Loaded"

set -eux

DATASET_CFG=$1   # eg training-data/lists/EXAMPLE_DIR
MODEL=$2         # eg inception_v3

MODEL_ID="${DATASET_CFG}__${MODEL}__COUNTS_MODE"

# move into yoloV5 directory
cd pytorch_classifier 

echo; echo "TRAINING START"
time python neuston_net.py TRAIN $MODEL  ../${DATASET_CFG}/training.txt ../${DATASET_CFG}/validation.txt ${MODEL_ID} --outdir ../training-output/{$MODEL_ID} --estop 20 --emax 200 --counts-mode


# run TEST labels if provided

if [ "$#" -eq 3 ]; then
    TESTDATA=$3
    
    echo; echo "TESTSET DETECT"    
    time python neuston_net.py RUN "../$TESTDATA" "../training-output/$MODEL_ID/$MODEL_ID.ptl" ${MODEL_ID}__TESTSET_RESULTS --outdir ../training-output/$MODEL_ID/testset_results --outfile img_results.csv
    
fi
   

echo; echo Job ID: $SLURM_JOB_ID is DONE!

