import json
import os
import re
from argparse import ArgumentParser
from collections import defaultdict

from dotenv import load_dotenv

from swe_bench.swebench.harness.run_evaluation import main as run_evaluation

load_dotenv()


def main(
	dataset_name: str,
	split: str,
	predictions_path: str,
	max_workers: int,
	run_id: str,
	output_file: str,
	timeout: int = 1_800,
) -> None:
	"""Evaluate predictions for a given tuned dataset.

	Args:
		dataset_name (str): Name of the dataset to evaluate.
		split (str): Split of the dataset to evaluate.
		predictions_path (str): Path to the file containing the
			predictions.
		max_workers (int): Maximum number of workers to use for
			evaluation.
		run_id (str): Unique identifier for the evaluation run.
		output_file (str): Path to the output file where results will be
			saved.
		timeout (int): Timeout for the evaluation run in seconds. By
			default, 1,800 seconds (30 minutes).
	"""
	tmp_dir = os.getenv("TMPDIR", "/tmp")

	predictions = defaultdict(list)
	with open(predictions_path, "r") as f:
		for line in f:
			data = json.loads(line)
			patch_pattern = r"(?:```(?:patch|diff).*?```|<patch>.*?</patch>)"

			try:
				match = re.search(patch_pattern, data["generation"], re.DOTALL)
				if match:
					data["generation"] = match.group(0)
			except TypeError:
				data["generation"] = ""
			except Exception as e:
				raise e

			predictions[data["num_files"]].append(data)

	results = {}
	for num_files, data in predictions.items():
		file = f"{tmp_dir}/pred_{run_id}_{num_files}.jsonl"

		with open(f"{tmp_dir}/pred_{run_id}_{num_files}.jsonl", "w") as f:
			for d in data:
				curr_d = {
					"generation": d["generation"],
					"instance_id": d["instance_id"],
				}
				f.write(json.dumps(curr_d) + "\n")

		report_file = run_evaluation(
			dataset_name,
			split=split,
			predictions_path=file,
			max_workers=max_workers,
			run_id=f"{run_id}_{os.path.basename(file).split('.')[0]}",
			cache_level="env",
			clean=False,
			timeout=timeout,
			force_rebuild=False,
			open_file_limit=4_096,
			instance_ids=[],
			out_dir=tmp_dir,
		)

		with open(report_file, "r") as f:
			results[num_files] = json.load(f)

		os.remove(file)

	with open(output_file, "w") as f:
		json.dump(results, f)


if __name__ == "__main__":
	parser = ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dataset_name",
		type=str,
		required=True,
		help="Name of the dataset to evaluate.",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="test",
		help="Split of the dataset to evaluate (default: 'test').",
	)
	parser.add_argument(
		"--predictions_path",
		type=str,
		required=True,
		help="Path to the file containing the predictions.",
	)
	parser.add_argument(
		"--max_workers",
		type=int,
		default=1,
		help="Maximum number of workers to use for evaluation.",
	)
	parser.add_argument(
		"--run_id",
		type=str,
		required=True,
		help="Unique identifier for the evaluation run.",
	)
	parser.add_argument(
		"--output_file",
		type=str,
		required=True,
		help="Path to the output file where results will be saved.",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=1_800,
		help="Timeout for the evaluation run in seconds (default: 1800).",
	)
	args = parser.parse_args()

	main(
		dataset_name=args.dataset_name,
		split=args.split,
		predictions_path=args.predictions_path,
		max_workers=args.max_workers,
		run_id=args.run_id,
		output_file=args.output_file,
		timeout=args.timeout,
	)
