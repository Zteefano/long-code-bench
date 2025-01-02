#!/bin/bash

#SBATCH --nodes=2                   # Numero di nodi
#SBATCH --ntasks-per-node=1         # Numero di GPU per nodo
#SBATCH --gpus-per-node=4           # GPU disponibili per nodo
#SBATCH --time=24:00:00             # Tempo massimo
#SBATCH --error=SWE_Eval.err        # File di errore
#SBATCH --output=SWE_Eval.out       # File di output
#SBATCH --partition=boost_usr_prod  # Partizione
#SBATCH --account=IscrC_TfG
#SBATCH --mem=64G

export HF_HOME=/leonardo/home/userexternal/asampier/IscrC_TfG/cache/cache
export MASTER_PORT=$((54000 + RANDOM % 1000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NODE_RANK=$SLURM_NODEID  # Rank del nodo

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"

srun torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK test.py