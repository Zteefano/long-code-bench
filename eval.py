import json
import os
from argparse import ArgumentParser
from typing import Literal, Optional

from datasets import load_from_disk
from dotenv import load_dotenv

from src.long_code_bench.inference.hf_eval import DatasetsEvaluator
from src.long_code_bench.models import APIModel, OpenSourceModel

load_dotenv()


def main(
	dataset_path: str,
	model_name: str,
	model_type: Literal["open", "api"],
	output: str,
	batch_size: Optional[int] = None,
	batch_queue: bool = True,
	open_kwargs_file: Optional[str] = None,
) -> None:
	"""Run evaluation on a dataset using a model.

	Args:
		dataset_path (str): The path to the dataset to evaluate. The
			dataset must be saved on disk.
		model_name (str): The name of the model to use for evaluation.
			If the modeli is an open-source model, the model name should
			be the model's path on the Hugging Face Hub.
		model_type (Literal["open", "api"]): The type of the model:
			'open' for open-source models, 'api' for API-based models.
		output (str): The path to save the evaluation output.
		batch_size (Optional[int], optional): The batch size to use for
			inference. By default, `None`. If `None`, the evalaution is
			not run in batches.
		batch_queue (bool, optional): Run evaluation using batch queue
			if model type is 'api'. Defaults to `True`.
		open_kwargs_file (Optional[str], optional): The path to a JSON
			file containing keyword arguments for the open-source model.
			Defaults to `None`. This argument allows for additional
			configuration of the open-source model like setting the
			`attn_implementation` or `torch_dtype`.
	"""
	dataset = load_from_disk(dataset_path)

	if model_type == "open":
		open_kwargs = {}
		if open_kwargs_file:
			with open(open_kwargs_file, "r") as file:
				open_kwargs = json.load(file)

		hf_token = os.getenv("HF_TOKEN", None)
		model = OpenSourceModel(model_name, token=hf_token, **open_kwargs)
	else:
		model = APIModel.from_model_name(model_name)

	evaluator = DatasetsEvaluator(
		model,
		dataset,
		"text",
		output,
		batch_size=batch_size,
		max_context_length=1_048_576,
		max_output_length=1_048_576,
	)

	if model_type == "api" and batch_queue:
		evaluator.run_batch_queue()
	else:
		evaluator.run()


if __name__ == "__main__":
	parser = ArgumentParser(
		description="Run evaluation on a dataset using a model."
	)
	parser.add_argument(
		"--dataset_path",
		type=str,
		help="The path to the dataset to evaluate. The dataset must be saved"
		+ " on disk.",
	)
	parser.add_argument(
		"--model_name",
		type=str,
		help="The name of the model to use for evaluation. If the modeli is"
		+ " an open-source model, the model name should be the model's path on"
		+ " the Hugging Face Hub",
	)
	parser.add_argument(
		"--model_type",
		type=str,
		choices=["open", "api"],
		help="The type of the model: 'open' for open-source models, 'api' for"
		+ " API-based models.",
	)
	parser.add_argument(
		"--output", type=str, help="The path to save the evaluation output."
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=None,
		help="The batch size to use for inference. By default, `None`. If"
		+ " `None`, the evalaution is not run in batches.",
	)
	parser.add_argument(
		"--batch_queue",
		type=bool,
		default=True,
		help="Run evaluation using batch queue if model type is 'api'.",
	)
	parser.add_argument(
		"--open_kwargs_file",
		type=str,
		default=None,
		help="The path to a JSON file containing keyword arguments for the"
		+ " open-source model. This argument allows for additional"
		+ " configuration of the open-source model like setting the"
		+ " `attn_implementation` or `torch_dtype`.",
	)
	args = parser.parse_args()

	main(
		args.dataset_path,
		args.model_name,
		args.model_type,
		args.output,
		args.batch_queue,
	)
