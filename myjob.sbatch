#!/bin/bash

#SBATCH --account=
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load anaconda
source ~/.bashrc
conda activate universal

cd $SLURM_SUBMIT_DIR

python train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml