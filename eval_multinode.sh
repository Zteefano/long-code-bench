#!/bin/bash
#SBATCH --out=multinode_accelerate.out
#SBATCH --err=multinode_accelerate.err
#SBATCH -A IscrC_TfG
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 2                # 1 node
##SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1 # 1 task
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4

export GPUS_PER_NODE=4
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export MASTER_PORT=6000
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_ADDR_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)
export BNB_CUDA_VERSION=121

export HF_HOME=/leonardo/home/userexternal/lromani0/IscrC_TfG/cache/cache

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo hostname = `hostname`
echo HOSTNAMES = $HOSTNAMES
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT
echo SLURM_PROCID= $SLURM_PROCID
echo NNODES= $NNODES    
echo WORLD_SIZE= $WORLD_SIZE    
echo NODE_RANK= $NODE_RANK      
echo NODE_NAME = $SLURMD_NODENAME
echo MASTER_ADDR_IP = $MASTER_ADDR_IP

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    --config_file config_accelerate.yaml
"
echo LAUNCHER=$LAUNCHER

module load cuda
module load gcc
source /leonardo/home/userexternal/lromani0/IscrC_TfG/.envs/lcb/bin/activate

srun $LAUNCHER test.py 