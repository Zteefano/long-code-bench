import json

from datasets import load_from_disk

from src.long_code_bench.inference.hf_eval import DatasetsEvaluator
from src.long_code_bench.models import OpenSourceModel

if __name__ == "__main__":
	dataset = load_from_disk(
		"data/swe_bench_small_text/SWE-bench__style-2__fs-all"
	)

	with open("keys.json", "r") as f:
		hf_key = json.load(f)["huggingface"]
	model = OpenSourceModel("ai21labs/Jamba-tiny-dev", token=hf_key)

	evaluator = DatasetsEvaluator(
		model,
		dataset,
		"text",
		"results/swe_bench_small.json",
		max_context_length=8_192,
		max_output_length=8_192,
	)
	evaluator.run()
