#!/bin/bash
#SBATCH --job-name=ppolog_sg_s20
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
ENV="SafetyPointGoal1-v0"

for SEED in 20; do
    echo "Starting seed $SEED"
    SAVE_DIR="/home/$USER/saferl_project/results/safetygym_seed${SEED}"
    $PYTHON ~/saferl_project/scripts/train_ppolog.py \
        --env $ENV \
        --seed $SEED \
        --total-steps 500000 \
        --cost-limit 25.0 \
        --device cuda:0 \
        --save-dir $SAVE_DIR
    echo "Finished seed $SEED"
done
