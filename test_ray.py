import ray
import time
import torch

# Optional: Ray will pick up environment variables (like CUDA_VISIBLE_DEVICES)
# But sometimes we want to enforce GPU usage explicitly:
# ray.init(address="auto", _temp_dir="/some/scratch/dir")

@ray.remote(num_gpus=1)
def test_gpu_task():
    # Just a simple check to see if we can create a CUDA tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((10, 10), device=device)
    return f"Device: {device}, Sum: {x.sum().item()}"

if __name__ == "__main__":
    # Connect to a Ray cluster that was started externally (by SLURM script)
    # using ray start --head ... & ray start --address ...
    print("Initializing Ray...")
    ray.init(address="auto")
    
    print("Ray cluster resources:", ray.cluster_resources())

    # Launch a few tasks to see if they schedule on multiple GPUs/nodes:
    futures = [test_gpu_task.remote() for _ in range(4)]
    results = ray.get(futures)
    
    print("Results from remote tasks:")
    for i, res in enumerate(results):
        print(f"Task {i} => {res}")

    # Keep the job alive for a bit in case you want to check Ray dashboard, etc.
    print("Sleeping for 30 seconds before exit...")
    time.sleep(30)
