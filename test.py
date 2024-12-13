import json

from datasets import load_from_disk

from src.long_code_bench.inference.hf_eval import DatasetsEvaluator
from src.long_code_bench.models import OpenSourceModel

if __name__ == "__main__":
	print('Loading dataset...')
	dataset = load_from_disk(
		"/leonardo/home/userexternal/asampier/IscrC_TfG/datasets/swebench_tuned_random"
		#"data/swe_bench_small_text/SWE-bench__style-2__fs-all"
	)

	print('Loading model...')
	#with open("keys.json", "r") as f:
	hf_key = 'NOT_USED' #json.load(f)["huggingface"]
	# "ai21labs/Jamba-tiny-dev"
	model = OpenSourceModel('/leonardo/home/userexternal/asampier/IscrC_TfG/cache/cache/hub/models--ai21labs--Jamba-tiny-dev/snapshots/ed303361004ac875426a61675edecf8e9d976882', token=hf_key)

	print('Initializing evaluator...')
	evaluator = DatasetsEvaluator(
		model,
		dataset,
		"text",
		"results/swe_bench_small.json",
		max_context_length=8_192,
		max_output_length=8_192,
	)
	evaluator.run()
