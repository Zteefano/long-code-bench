import json
from typing import Generator, List, Optional

import datasets as dts
from dotenv import load_dotenv
from tqdm.auto import tqdm

from src.long_code_bench.models import Model

load_dotenv()


def _iterate_dataset(
	dataset: dts.Dataset, batch_size: Optional[int] = None
) -> Generator[dict, None, None]:
	if batch_size is None:
		for instance in dataset:
			yield dict(instance)
	else:
		len_dataset = len(dataset)
		for i in range(0, len_dataset, batch_size):
			yield dataset[i : min(i + batch_size, len_dataset)]


class DatasetsEvaluator:
	"""Class for running inference on Hugging Face datasets.

	This class takes a model and a dataset loaded through the Hugging
	Face datasets library and runs inference on the dataset, providing
	a generation for each instance in the dataset.

	Args:
		model (Model): The model to use for inference.
		dataset (dts.Dataset | dts.DatasetDict): The dataset to run
			inference on.
		prompt_feature (str): The feature in the dataset that
			corroponds to the prompt.
		results_file (str): The file where to store the results.
		splits (List[str]): The splits to use from the dataset. By
			default, `None`. Only used if `dataset` is a `DatasetDict`.
		max_context_length (Optional[int]): The maximum length of the
			context to provide to the model. By default, `None`. If
			`None`, contexts of any length are processed.
		max_output_length (Optional[int]): The maximum length of the
			generated text. By default, `None`. If `None`, no maximum
			length is enforced.
		batch_size (Optional[int]): The batch size to use for inference.
			By default, `16`. If `None`, the evalaution is not run in
			batches.
	"""

	def __init__(
		self,
		model: Model,
		dataset: dts.Dataset | dts.DatasetDict,
		prompt_feature: str,
		results_file: str,
		splits: Optional[List[str]] = None,
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
		batch_size: Optional[int] = 16,
	) -> None:
		self.model = model
		self.max_context_length = max_context_length
		self.max_output_length = max_output_length
		self.batch_size = batch_size

		self.dataset = dataset
		if isinstance(dataset, dts.DatasetDict) and splits is not None:
			self.dataset = dataset[splits]

		self.prompt_feature = prompt_feature
		self.results_file = results_file

	def run(self) -> None:
		"""Run inference on the dataset."""
		open(self.results_file, "w").close()
		bar = tqdm(total=self._len_dataset(), desc="Processing instances")
		iterator = (
			self._iterate_dataset_batch()
			if self.batch_size
			else self._iterate_dataset()
		)
		for instance in iterator:
			self._process_instance(instance)
			bar.update(len(instance["instance_id"]) if self.batch_size else 1)
		bar.close()

	def run_batch_queue(self, file_name: Optional[str] = None) -> None:
		"""Run inference on the dataset using batch processing.

		This is the version of `run` that queues completion
		requests for each instance in the dataset to be completed
		asynchronously on the respective provider's platform.

		Args:
			file_name (Optional[str], optional): The file to store the
				requests to be processed. If `None`, a temporary file is
				used. Defaults to `None`.
		"""
		tasks = []
		for idx, instance in enumerate(self._iterate_dataset()):
			tasks.append(
				{
					"prompt": instance[self.prompt_feature],
					"id": f"{instance['instance_id']}-{idx}",
					"instance_id": instance["instance_id"],
					"num_files": instance["num_files"],
					"num_tokens": instance["num_tokens"],
				}
			)

		results = self.model.generate_batch(
			[task["prompt"] for task in tasks],
			max_context_length=self.max_context_length,
			max_output_length=self.max_output_length,
			ids=[task["id"] for task in tasks],
			file_name=file_name,
			batch_size=self.batch_size,  # type: ignore
		)

		with open(self.results_file, "w") as f:
			for result, task in zip(results, tasks, strict=True):
				f.write(
					json.dumps(
						{
							"prompt": task["prompt"],
							"generation": result,
							"instance_id": task["instance_id"],
							"num_files": task["num_files"],
							"num_tokens": task["num_tokens"],
						}
					)
					+ "\n"
				)

	def _process_instance(self, batch: dict) -> None:
		prompt = batch[self.prompt_feature]

		if self.batch_size is None:
			generation = self.model.generate(
				prompt, self.max_context_length, self.max_output_length
			)
			prompt = [prompt]
			generation = [generation]
			ids = [batch["instance_id"]]
			num_files = [batch["num_files"]]
			num_tokens = [batch["num_tokens"]]
		else:
			generation = self.model.generate_batch(
				prompt,
				max_context_length=self.max_context_length,
				max_output_length=self.max_output_length,
			)
			ids = batch["instance_id"]
			num_files = batch["num_files"]
			num_tokens = batch["num_tokens"]

		for i, (instance, gen) in enumerate(
			zip(ids, generation, strict=False)
		):
			to_write = {
				"prompt": prompt[i],
				"generation": gen,
				"instance_id": instance,
				"num_files": num_files[i],
				"num_tokens": num_tokens[i],
			}

			with open(self.results_file, "a") as f:
				f.write(json.dumps(to_write) + "\n")

	def _iterate_dataset(self) -> Generator[dict, None, None]:
		if isinstance(self.dataset, dts.DatasetDict):
			for split in self.dataset:
				for instance in _iterate_dataset(self.dataset[split]):
					yield instance
		elif isinstance(self.dataset, dts.Dataset):
			for instance in _iterate_dataset(self.dataset):
				yield instance

	def _iterate_dataset_batch(self) -> Generator[dict, None, None]:
		if isinstance(self.dataset, dts.DatasetDict):
			for split in self.dataset:
				for batch in _iterate_dataset(
					self.dataset[split], self.batch_size
				):
					yield batch
		elif isinstance(self.dataset, dts.Dataset):
			for batch in _iterate_dataset(self.dataset, self.batch_size):
				yield batch

	def _len_dataset(self) -> int:
		if isinstance(self.dataset, dts.DatasetDict):
			return sum(len(self.dataset[split]) for split in self.dataset)
		elif isinstance(self.dataset, dts.Dataset):
			return len(self.dataset)
		else:
			raise ValueError("Dataset must be a Dataset or DatasetDict.")
