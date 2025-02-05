#!/bin/bash
#SBATCH --out=multinode_accelerate.out
#SBATCH --err=multinode_accelerate.err
#SBATCH -A IscrC_TfG
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 2                # 1 node
##SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1 # 1 task
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4

set -x -e

######################
### Set enviroment ###
######################
module load cuda
module load gcc
source /leonardo/home/userexternal/lromani0/IscrC_TfG/.envs/lcb/bin/activate
export GPUS_PER_NODE=4
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
#  export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"

export LAUNCHER="accelerate launch \
    --config_file configs/fsdp_config.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 6000 \
    "
export SCRIPT="${ACCELERATE_DIR}/examples/complete_nlp_example.py"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT" 
srun $CMD