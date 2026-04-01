#!/bin/bash
#SBATCH --job-name=speedtest
#SBATCH --account=ece567w26_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/saferl_project/logs/speedtest_%j.out
#SBATCH --error=/home/%u/saferl_project/logs/speedtest_%j.err

export MUJOCO_GL=egl
PYTHON=/home/dchance/.conda/envs/saferl/bin/python

$PYTHON ~/saferl_project/scripts/train_ppolog.py \
    --env SafetyPointGoal1-v0 \
    --seed 0 \
    --total-steps 90000 \
    --device cuda:0 \
    --save-dir /home/$USER/saferl_project/results/speedtest
