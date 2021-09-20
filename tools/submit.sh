#!/bin/bash

#SBATCH -N 1
#SBATCH -M swarm
#SBATCH -p gpu
#SBATCH -n 5
#SBATCH --gres=gpu:4

export PYTHONUNBUFFERED=1
python train.py --train-json $1 --lr 0.01 --train-batch 128 --epochs 50 --schedule 25 40 --gamma 0.1 --wd 1e-3
