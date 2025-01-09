# export HF_HOME=/leonardo/home/userexternal/asampier/IscrC_TfG/cache/cache
print('Loading libraries...')
import json

from datasets import load_from_disk

from src.long_code_bench.inference.hf_eval import DatasetsEvaluator
from src.long_code_bench.models import OpenSourceModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":

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

	model = OpenSourceModel("ai21labs/AI21-Jamba-1.5-Mini", token=hf_key)

	print('Initializing evaluator...')
	evaluator = DatasetsEvaluator(
		model,
		dataset,
		"text",
		"results/swe_bench_ver_tuned_small.json",
		max_context_length=80_000, #1_000_000, #8_192,
		max_output_length=80_000, #1_000_000, #8_192,
	)
	evaluator.run()
