#!/bin/bash
#SBATCH --job-name=ppolog_scg_s0
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

echo "Starting seed 0"
SAVE_DIR="/home/$USER/saferl_project/results/scg_seed0_2M"
$PYTHON ~/saferl_project/scripts/train_ppolog.py     --env SCG-CartPole-v0     --seed 0     --total-steps 2000000     --cost-limit 25.0     --device cuda:0     --save-dir $SAVE_DIR
echo "Finished seed 0"
