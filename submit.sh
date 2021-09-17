#!/bin/bash

#SBATCH -N 1
#SBATCH -M swarm
#SBATCH -p gpu
#SBATCH -n 5
#SBATCH --gres=gpu:4

export PYTHONUNBUFFERED=1
python train.py --train-json $1 --lr 0.1 --train-batch 64 --epochs 30 --schedule 15 25 --gamma 0.1 --wd 1e-4
