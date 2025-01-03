#!/bin/bash

#SBATCH --nodes=2                   # Numero di nodi
#SBATCH --ntasks-per-node=1      # Numero di processi per nodo
#SBATCH --gpus-per-node=4 
#SBATCH --cpus-per-task=1          ### Number of threads per task (OMP threads)
#SBATCH --time=24:00:00             # Tempo massimo
#SBATCH --error=SWE_Eval.err        # File di errore
#SBATCH --output=SWE_Eval.out       # File di output
#SBATCH --partition=boost_usr_prod  # Partizione
#SBATCH --gres=gpu:4
#SBATCH --account=IscrC_TfG
#SBATCH --mem=64G

#### SBATCH --gpus-per-task=1
#####SBATCH --ntasks=2         # Numero di GPU per nodo
###SBATCH --gpus-per-node=4           # GPU disponibili per nodo

#nodes = ($(scontrol show hostnames $SLURM_JOB_NODELIST))
#nodes_array = ($nodes)
#head_node = ${nodes_array[0]}
#head_node_ip = $(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)

export HF_HOME=/leonardo/home/userexternal/asampier/IscrC_TfG/cache/cache

# Configurazione ambiente
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
#export MASTER_PORT=45216
export MASTER_PORT=$(shuf -i 10000-20000 -n 1)
export WORLD_SIZE=8 #$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID

echo "NODELIST="${SLURM_NODELIST}
echo "MASTER_ADDR="$MASTER_ADDR
echo "WORLD_SIZE="$WORLD_SIZE
echo "LOCAL_RANK="$LOCAL_RANK

#echo "Head node: $head_node_ip"
#export LOGLEVEL=INFO

srun torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --rdzv-id=$JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR test.py


#srun python test.py

# --nproc_per_node=8 \ # is --gpus-per-task
#srun torchrun --nnodes=2 --nproc_per_node=8  test.py



#export MASTER_PORT=$((54000 + RANDOM % 1000))
#export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export NODE_RANK=$SLURM_NODEID  # Rank del nodo
#export WORLD_SIZE=$SLURM_NTASKS
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
