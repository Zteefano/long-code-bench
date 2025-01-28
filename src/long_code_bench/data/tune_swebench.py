import itertools
import os
import pathlib
import random
from argparse import ArgumentParser
from copy import deepcopy
from typing import List, Literal, Optional

import datasets as dts
from dotenv import load_dotenv
from tqdm.auto import tqdm

from long_code_bench.data.count_tokens import count_tokens
from swe_bench.swebench.inference.make_datasets.bm25_retrieval import (
	main as bm25_main,
)
from swe_bench.swebench.inference.make_datasets.create_instance import (
	add_text_inputs,
)
from swe_bench.swebench.inference.make_datasets.create_text_dataset import (
	extract_fields,
)

load_dotenv()


def _process_retrieval_file(dataset: str, splits: List[str]) -> str:
	tmp_dir = os.getenv("TMPDIR", "/tmp")
	bm25_main(
		dataset,
		document_encoding_style="file_name_and_contents",
		output_dir=tmp_dir,
		splits=splits,
		shard_id=None,
		num_shards=20,
		leave_indexes=False,
	)
	if pathlib.Path(dataset).exists():
		data_name = os.path.basename(dataset)
	else:
		data_name = dataset.replace("/", "__")
	return f"{tmp_dir}/{data_name}/file_name_and_contents.retrieval.jsonl"


def _save_dataset(
	data: dts.DatasetDict,
	output_dir: Optional[str] = None,
	hfhub_dataset: Optional[str] = None,
) -> None:
	if output_dir:
		data.save_to_disk(output_dir)
	if hfhub_dataset:
		token = os.getenv("HF_TOKEN_WRITE")
		if not token:
			raise ValueError(
				"The Hugging Face token is required to push to the hub."
			)
		data.push_to_hub(hfhub_dataset, token=token)


def make_tunable_swebench(
	dataset: str,
	splits: List[str],
	prompt_style: Literal["style-2", "style-3", "full_file_gen"],
	retrieval_type: Literal["bm25", "random"],
	max_k: int,
	max_tokens: Optional[int] = None,
	retrieval_file: Optional[str] = None,
	output_dir: Optional[str] = None,
	hfhub_dataset: Optional[str] = None,
	num_workers: int = 1,
) -> None:
	"""Create a tunable version of the SWE-Bench dataset.

	A tunable version of the SWE-Bench dataset is created by following
	the same steps as the original dataset creation script, but storing,
	for each problem statement, a number of instances varying based on
	the amount of retrived files.

	In SWE-Bench, for each problem statement, models are asked to solve
	the issue by observing a subset of the files in the repository. This
	function creates a dataset where the number of files to observe is
	a parameter that can be tuned.

	Such dataset should allow to evaluate the performance of models in
	a more fine-grained way, by observing the impact of the context
	size on the performance of the model.

	Args:
		dataset (str): Either the name of the dataset from the Hugging
			Face Hub or the path to the dataset on disk.
		splits (List[str]): The splits to use from the dataset.
		prompt_style (Literal["style-2", "style-3", "full_file_gen"]):
			The style of prompt to generate for the dataset. Refer to
			[this README](https://github.com/princeton-nlp/SWE-bench/blob/main/swebench/inference/make_datasets/README.md)
			for more information.
		retrieval_type (Literal["bm25", "random"]): The type of
			retrieval strategy to use. If `bm25`, the BM25 retrieval
			results will be computed. If `random`, random files will be
			retrieved together with the oracle ones.
		max_k (int): The maximum number of files to retrieve for each
			problem statement if `max_tokens` is `None`, or the maximum
			number of samples to generate per issue, divided evenly by
			number of tokens in the context if `max_tokens` is not
			`None`.
		max_tokens (Optional[int]): The maximum number of tokens to
			consider when creating the dataset. If `None`, no limit is
			set. Defaults to `None`. Notice that this paramter changes
			the behavior of `max_k`, as it stops referring to the number
			of files to retrieve, but to the number samples to generate
			per issue, divided evenly by number of tokens in the
			context.
		retrieval_file (Optional[str]): The path to the file where the
			BM25 retrieval results are stored. If `None`, the retrieval
			results will be computed. Defaults to `None`.
		output_dir (Optional[str]): The path to the output directory. If
			`None`, the dataset will not be saved to disk. Defaults to
			`None`.
		hfhub_dataset (Optional[str]): The name of the dataset to push
			to the Hugging Face Hub. If `None`, the dataset will not be
			pushed to the Hub.
		num_workers (int): The number of workers to use when counting
			the number of tokens for each instance. Defaults to `1`.

	Raises:
		ValueError: If the dataset is not an instance of `DatasetDict`.
	"""
	data: dts.DatasetDict
	if pathlib.Path(dataset).exists():
		res = dts.load_from_disk(dataset)
	else:
		res = dts.load_dataset(dataset)
	if not isinstance(res, dts.DatasetDict):
		raise ValueError("The dataset must be a DatasetDict.")
	data = res

	retrieval_file = (
		_process_retrieval_file(dataset, splits)
		if retrieval_file is None and retrieval_type == "bm25"
		else retrieval_file
	)

	split_instances = {}
	for split, k in tqdm(
		itertools.product(splits, range(max_k)),
		total=len(splits) * max_k,
		desc="Retrieving files",
	):
		split_instances[f"{split}-{k}"] = {
			x["instance_id"]: deepcopy(x)  # type: ignore
			for x in data[split]
		}
		add_text_inputs(
			split_instances[f"{split}-{k}"],
			retrieval_file,
			k,
			prompt_style,
			max_tokens=(max_tokens // max_k) * k if max_tokens else None,
			file_source=f"oracle+{retrieval_type}",
		)

	columns = [
		"instance_id",
		"num_files",
		"retrieval_strategy",
		"text",
		"repo",
		"base_commit",
		"problem_statement",
		"hints_text",
		"created_at",
		"patch",
		"test_patch",
		"version",
		"FAIL_TO_PASS",
		"PASS_TO_PASS",
		"environment_setup_commit",
	]
	split_data = {}
	for split in splits:
		split_data[split] = {key: [] for key in columns}
		for instance, k in tqdm(
			itertools.product(data[split], range(max_k)),
			total=len(data[split]) * max_k,
			desc=f"Processing {split} split.",
		):
			datum = extract_fields(
				split_instances[f"{split}-{k}"][instance["instance_id"]]  # type: ignore
			)
			if datum is None:
				continue
			datum["num_files"] = k
			datum["retrieval_strategy"] = retrieval_type
			for key in columns:
				split_data[split][key].append(datum[key])
		split_data[split] = dts.Dataset.from_dict(split_data[split])

	data_final = dts.DatasetDict(split_data)
	data_final = count_tokens(data_final, num_workers=num_workers)
	_save_dataset(data_final, output_dir, hfhub_dataset)


if __name__ == "__main__":
	parser = ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dataset",
		type=str,
		default="princeton-nlp/SWE-bench",
		help="Dataset's name from the Hugging Face Hub or path to a directory"
		+ " where the dataset is stored.",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=["dev", "test"],
		help="Splits to use from the dataset.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Path to the output directory.",
	)
	parser.add_argument(
		"--retrieval_file",
		type=str,
		default=None,
		help="Path to the file where the BM25 retrieval results are stored.",
	)
	parser.add_argument(
		"--retrieval_type",
		type=str,
		default="bm25",
		choices=["bm25", "random"],
		help="The type of retrieval strategy to use.",
	)
	parser.add_argument(
		"--prompt_style",
		type=str,
		default="style-2",
		choices=["style-2", "style-3", "full_file_gen"],
		help="The style of prompt to generate for the dataset.",
	)
	parser.add_argument(
		"--max_k",
		type=int,
		help="The maximum number of files to retrieve per problem statement"
		+ " or the maximum number of samples to generate per issue.",
	)
	parser.add_argument(
		"--max_tokens",
		type=int,
		default=None,
		help="The maximum number of tokens to consider when creating the"
		+ " dataset.",
	)
	parser.add_argument(
		"--hfhub_dataset",
		type=str,
		default=None,
		help="Name of the dataset to push to the Hugging Face Hub.",
	)
	parser.add_argument(
		"--num_workers",
		type=int,
		default=1,
		help="The number of workers to use when counting tokens per instance.",
	)
	parser.add_argument(
		"--random_seed",
		type=int,
		default=18_249_170_682_302,
		help="Random seed for reproducibility.",
	)
	args = parser.parse_args()

	random.seed(args.random_seed)
	make_tunable_swebench(
		**{k: v for k, v in vars(args).items() if k != "random_seed"}
	)
