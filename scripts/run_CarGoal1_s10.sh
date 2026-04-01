#!/bin/bash
#SBATCH --job-name=ppolog_CarGoal1_s10
#SBATCH --account=ece567w26_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/saferl_project/logs/%x_%j.out
#SBATCH --error=/home/%u/saferl_project/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dchance@umich.edu

export MUJOCO_GL=egl
PYTHON=/home/dchance/.conda/envs/saferl/bin/python

echo "Starting SafetyCarGoal1-v0 seed 10"
SAVE_DIR="/home/$USER/saferl_project/results/CarGoal1_s10"
$PYTHON ~/saferl_project/scripts/train_ppolog.py     --env SafetyCarGoal1-v0     --seed 10     --total-steps 2500000     --device cuda:0     --save-dir $SAVE_DIR
echo "Finished SafetyCarGoal1-v0 seed 10"
