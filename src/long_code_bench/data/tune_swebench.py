import itertools
import json
import os
import pathlib
import shutil
from argparse import ArgumentParser
from copy import deepcopy
from typing import Callable, List, Literal, Optional, Tuple

import datasets as dts
from tqdm.auto import tqdm

from swe_bench.swebench.inference.make_datasets.bm25_retrieval import (
	main as bm25_main,
)
from swe_bench.swebench.inference.make_datasets.create_instance import (
	add_text_inputs,
)
from swe_bench.swebench.inference.make_datasets.create_text_dataset import (
	extract_fields,
)


def _process_retrieval_file(
	dataset: str, retrieval_file: str, splits: List[str]
) -> Tuple[str, Callable[[], None]]:
	if not pathlib.Path(retrieval_file).exists():
		tmp_dir = os.getenv("TMPDIR", "/tmp")
		bm25_main(
			dataset,
			document_encoding_style="file_name_and_contents",
			output_dir=tmp_dir,
			splits=splits,
			shard_id=None,
			num_shards=20,
			leave_indexes=True,
		)
		data_name = os.path.basename(dataset)
		return (
			f"/{tmp_dir}/{data_name}/file_name_and_contents.retrieval.jsonl",
			lambda: shutil.rmtree(f"/{tmp_dir}/{data_name}"),
		)
	return retrieval_file, lambda: None


def make_tunable_swebench(
	dataset: str,
	splits: List[str],
	output_dir: str,
	retrieval_file: str,
	prompt_style: Literal["style-2", "style-3", "full_file_gen"],
	max_k: int,
	hfhub_dataset: Optional[str] = None,
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
		output_dir (str): The directory where to store the dataset.
		retrieval_file (str): The file containing results from the BM25
			retrieval. If the file does not exist, it will be created by
			running the BM25 retrieval.
		prompt_style (Literal["style-2", "style-3", "full_file_gen"]):
			The style of prompt to generate for the dataset. Refer to
			[this README](https://github.com/princeton-nlp/SWE-bench/blob/main/swebench/inference/make_datasets/README.md)
			for more information.
		max_k (int): The maximum number of files to retrieve for each
			problem statement.
		hfhub_dataset (Optional[str]): The name of the dataset to push
			to the Hugging Face Hub. If `None`, the dataset will not be
			pushed to the Hub.

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

	retrieval_file, handle_retr = _process_retrieval_file(
		dataset, retrieval_file, splits
	)

	split_instances = {}
	for split, k in tqdm(
		itertools.product(splits, range(max_k)),
		total=len(splits) * max_k,
		desc="Retrieving files.",
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
			file_source="oracle+bm25",
		)

	handle_retr()

	columns = [
		"instance_id",
		"num_bm_files",
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
			datum["num_bm_files"] = k
			for key in columns:
				split_data[split][key].append(datum[key])
		split_data[split] = dts.Dataset.from_dict(split_data[split])

	data_final = dts.DatasetDict(split_data)
	data_final.save_to_disk(output_dir)

	if hfhub_dataset:
		try:
			token = json.load(open("keys.json"))["huggingface_write"]
		except (FileNotFoundError, KeyError):
			raise ValueError(
				"The Hugging Face token is required to push to the hub."
			) from None
		data_final.push_to_hub(hfhub_dataset, token=token)


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
		default=["train", "test"],
		help="Splits to use from the dataset.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		help="Path to the output directory.",
	)
	parser.add_argument(
		"--retrieval_file",
		type=str,
		help="Path to the file where the BM25 retrieval results are stored.",
	)
	parser.add_argument(
		"--prompt_style",
		type=str,
		default="style-3",
		choices=["style-2", "style-3", "full_file_gen"],
		help="The style of prompt to generate for the dataset.",
	)
	parser.add_argument(
		"--max_k",
		type=int,
		help="The maximum number of files to retrieve per problem statement.",
	)
	parser.add_argument(
		"--hfhub_dataset",
		type=str,
		default=None,
		help="Name of the dataset to push to the Hugging Face Hub.",
	)
	make_tunable_swebench(**vars(parser.parse_args()))
