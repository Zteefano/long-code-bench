#!/bin/bash
        
#SBATCH --job-name=Jamba_on_ray
#SBATCH --time=04:00:00
#SBATCH --account=IscrC_TfG
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --error llm_launcher-%j.err
#SBATCH --output llm_launcher-%j.out

# Script to launch a containerized version of RAY and vllm.

# API Token used by vllm to secure the requests
API_TOKEN="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"
# MODEL_PATH="/leonardo/home/userexternal/lromani0/IscrC_TfG/cache/cache/hub/models--ai21labs--AI21-Jamba-1.5-Mini/snapshots/1840d3373c51e4937f4dbaaaaf8cac1427b46858"

MODEL_PATH="/leonardo_scratch/large/userinternal/mviscia1/models/Llama-3.1_405B-Instruct"

# You can build an image using the script container/build.sh, or you can use the one provided in this folder
IMAGE_NAME="vllm_singularity-2025-01-14-2e6300811da9.sif"
IMAGE_PATH="/leonardo/home/userexternal/lromani0/IscrC_TfG/docker_images/$IMAGE_NAME"

export NCCL_IB_HCA="mlx5" # RDMA interfaces to use for communication
export NCCL_SOCKET_IFNAME="ib0"

# Get the worker list associated to this slurm job
worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# Set the first worker as the head node and get his ip
head_node=${worker_list[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start the head node
echo [INFO]: Starting head node $head_node at $head_node_ip

# Create overlay for the head node
echo [INFO]: Removing old headnode overlay and creating a new writable layer
OVERLAY_PATH="${SCRATCH}/worker_overlay_0.img"
rm $OVERLAY_PATH
singularity overlay create --sparse --size 16000 $OVERLAY_PATH

# Define commands to be runned on the head node
ray_start_head_command="ray start --head --include-dashboard True --dashboard-port 8265 --dashboard-host="0.0.0.0" --node-ip-address="$head_node_ip" --port=6379 --num-cpus $SLURM_CPUS_PER_TASK --disable-usage-stats --block"
vllm_serve_command="vllm serve $MODEL_PATH --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --pipeline-parallel-size $SLURM_JOB_NUM_NODES --api-key $API_TOKEN"

#Run the commands. Note that ntasks in srun is set to 1 because otherwise the ray start command would be executing ntasks times, effectively spawning ntasks*--num-cpus workers.
echo "[INFO]: srun --nodes=1 --ntasks=1 -w $head_node singularity exec --nv --bind /leonardo_scratch:/leonardo_scratch,/leonardo_work:/leonardo_work --overlay $OVERLAY_PATH $IMAGE_PATH bash -c \"$ray_start_head_command & sleep 60 && $vllm_serve_command\""
srun --nodes=1 --ntasks=1 -w $head_node singularity exec --nv --bind /leonardo_scratch:/leonardo_scratch,/leonardo_work:/leonardo_work --overlay $OVERLAY_PATH $IMAGE_PATH bash -c "$ray_start_head_command & sleep 60 && $vllm_serve_command" &

# Print ssh tunnel instruction. This is needed to access the ray web ui.
ray_dashboard_port=$((1024 + $RANDOM % 64511))
echo ===================================================
echo [INFO]: To access the RAY web ui, remember to open a ssh tunnel with: 
echo ssh -L $ray_dashboard_port:$head_node_ip:8265 ${USER}@login02-ext.leonardo.cineca.it -N
echo then you can connect to the dashboard at http://127.0.0.1:$ray_dashboard_port
echo ===================================================

sleep 5

# Start the workers
worker_num=$((SLURM_JOB_NUM_NODES - 1))

# Define worker node startup command. Here we don't need to startup the vllm server.
ray_start_worker_command="ray start --address "$head_node_ip:6379" --num-cpus $SLURM_CPUS_PER_TASK --disable-usage-stats --block"

for ((i = 1; i <= worker_num; i++)); do
    worker_node_i=${worker_list[$i]}
    worker_node_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node_i" hostname --ip-address)

    echo [INFO]: Removing old workernode ${i} overlay and creating a new writable layer
    OVERLAY_PATH="${SCRATCH}/worker_overlay_${i}.img"
    rm $OVERLAY_PATH
    singularity overlay create --sparse --size 16000 $OVERLAY_PATH

    echo [INFO]: Starting worker $i $worker_node_i at $worker_node_ip
    echo [INFO]: srun --nodes=1 --ntasks=1 -w $worker_node_i singularity exec --nv --bind /leonardo_scratch:/leonardo_scratch,/leonardo_work:/leonardo_work --overlay $OVERLAY_PATH $IMAGE_PATH $ray_start_worker_command
    srun --nodes=1 --ntasks=1 -w $worker_node_i singularity exec --nv --bind /leonardo_scratch:/leonardo_scratch,/leonardo_work:/leonardo_work --overlay $OVERLAY_PATH $IMAGE_PATH $ray_start_worker_command &
done

sleep 10

echo [INFO]: everything is running.

sleep infinity
