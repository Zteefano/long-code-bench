# export HF_HOME=/leonardo/home/userexternal/asampier/IscrC_TfG/cache/cache
import json

from datasets import load_from_disk
import torch.distributed as dist

from src.long_code_bench.inference.hf_eval import DatasetsEvaluator
from src.long_code_bench.models import OpenSourceModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from mpi4py import MPI

MPI.COMM_WORLD.Barrier()


# Inizializza il processo
def init_process():
	rank = int(os.environ["RANK"])
	world_size = int(os.environ["WORLD_SIZE"])
	dist.init_process_group(
    backend="nccl",
    init_method="env://"
	)
	torch.cuda.set_device(dist.get_rank())
	print('World size:', world_size, 'Rank:', rank)


if __name__ == "__main__":
	print('Starting...')
	init_process()
	#dist.barrier()
	print('Process initialized')


	print('Available devices:', torch.cuda.device_count())

	print('Loading dataset...')
	dataset = load_from_disk(
		"/leonardo/home/userexternal/asampier/IscrC_TfG/datasets/swebench_ver_tuned_small"
		#"data/swe_bench_small_text/SWE-bench__style-2__fs-all"
	)

	print('Loading model...')
	#with open("keys.json", "r") as f:
	hf_key = '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' #json.load(f)["huggingface"]
	# "ai21labs/Jamba-tiny-dev"
	cache = '/leonardo/home/userexternal/asampier/IscrC_TfG/cache/cache/hub/'
	#model = OpenSourceModel(cache+'models--ai21labs--Jamba-tiny-dev/snapshots/ed303361004ac875426a61675edecf8e9d976882', token=hf_key)
	#model = OpenSourceModel(cache+'models--ai21labs--AI21-Jamba-1.5-Mini', token=hf_key)
	# ai21labs/Jamba-tiny-dev
	# ai21labs/AI21-Jamba-1.5-Mini
	model = OpenSourceModel("ai21labs/Jamba-tiny-dev", token=hf_key)

	print('Model loaded')

	

	print('Initializing evaluator...')
	evaluator = DatasetsEvaluator(
		model,
		dataset,
		"text",
		"results/swe_bench_ver_tuned_small.json",
		max_context_length=1_000_000, #8_192,
		max_output_length=1_000_000, #8_192,
		#gpu_id=gpu_id, global_rank=global_rank,
	)
	evaluator.run()

	# Cleanup
	dist.destroy_process_group()