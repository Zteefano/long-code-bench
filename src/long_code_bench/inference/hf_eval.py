import json
from typing import Generator, List, Optional

import datasets as dts
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.long_code_bench.models import Model


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

	def _process_instance(self, batch: dict) -> None:
		prompt = batch[self.prompt_feature]

		if self.batch_size is None:
			generation = self.model.generate(
				prompt, self.max_context_length, self.max_output_length
			)
			prompt = [prompt]
			generation = [generation]
			ids = [batch["instance_id"]]
		else:
			generation = self.model.generate_batch(
				prompt,
				max_context_length=self.max_context_length,
				max_output_length=self.max_output_length,
			)
			ids = batch["instance_id"]

		for i, (instance, gen) in enumerate(
			zip(ids, generation, strict=False)
		):
			to_write = {
				"prompt": prompt[i],
				"generation": gen,
				"instance_id": instance,
			}

			with open(self.results_file, "a") as f:
				f.write(json.dumps(to_write) + "\n")

	def _iterate_dataset(self) -> Generator[dict, None, None]:
		if isinstance(self.dataset, dts.DatasetDict):
			for split in self.dataset:
				for instance in self.dataset[split]:
					yield dict(instance)
		elif isinstance(self.dataset, dts.Dataset):
			for instance in self.dataset:
				yield dict(instance)

	def _iterate_dataset_batch(self) -> Generator[dict, None, None]:
		if isinstance(self.dataset, dts.DatasetDict):
			for split in self.dataset:
				dataloader = DataLoader(
					self.dataset[split],  # type: ignore
					batch_size=self.batch_size,
				)
				for batch in dataloader:
					yield batch
		elif isinstance(self.dataset, dts.Dataset):
			dataloader = DataLoader(self.dataset, batch_size=self.batch_size)  # type: ignore
			for batch in dataloader:
				yield batch

	def _len_dataset(self) -> int:
		if isinstance(self.dataset, dts.DatasetDict):
			return sum(len(self.dataset[split]) for split in self.dataset)
		elif isinstance(self.dataset, dts.Dataset):
			return len(self.dataset)
		else:
			raise ValueError("Dataset must be a Dataset or DatasetDict.")
