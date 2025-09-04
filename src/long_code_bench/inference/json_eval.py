import json
from typing import Dict, Generator, List, Optional

from tqdm.auto import tqdm

from src.long_code_bench.models import Model


def _iterate_dataset(
	dataset: List[Dict[str, str]], batch_size: Optional[int] = None
) -> Generator[dict, None, None]:
	if batch_size is None:
		for instance in dataset:
			yield instance
	else:
		len_dataset = len(dataset)
		for i in range(0, len_dataset, batch_size):
			current = dataset[i : min(i + batch_size, len_dataset)]
			keys = current[0].keys()
			yield {k: [d[k] for d in current] for k in keys}


class JSONEvaluator:
	"""Class for running inference on JSON datasets.

	This class takes a model and a dataset loaded from a JSON file and
	runs inference on the dataset, providing a generation for each
	instance in the dataset.

	Args:
		model (Model): The model to use for inference.
		dataset (List[dict]): The dataset to run inference on.
		prompt_feature (str): The feature in the dataset that
			corresponds to the prompt.
		results_file (str): The file where to store the results.
		max_context_length (Optional[int]): The maximum length of the
			context to provide to the model. By default, `None`. If
			`None`, contexts of any length are processed.
		max_output_length (Optional[int]): The maximum length of the
			generated text. By default, `None`. If `None`, no maximum
			length is enforced.
		batch_size (Optional[int]): The batch size to use for inference.
			By default, `16`. If `None`, the evaluation is not run in
			batches.
	"""

	def __init__(
		self,
		model: Model,
		dataset: List[Dict[str, str]],
		prompt_feature: str,
		results_file: str,
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
		batch_size: Optional[int] = 16,
	) -> None:
		self.model = model
		self.dataset = dataset
		self.prompt_feature = prompt_feature
		self.results_file = results_file
		self.max_context_length = max_context_length
		self.max_output_length = max_output_length
		self.batch_size = batch_size

	def run(self) -> None:
		"""Run inference on the dataset."""
		bar = tqdm(total=len(self.dataset), desc="Processing instances")
		for instance in _iterate_dataset(self.dataset, self.batch_size):
			self._process_instance(instance)
			bar.update(len(instance) if self.batch_size else 1)
		bar.close()

		to_write = []
		with open(self.results_file, "r") as f:
			for idx, line in enumerate(f):
				curr_data = json.loads(line)
				to_write.append(
					{
						"text": curr_data["generation"],
						"id": str(idx),
					}
				)
		with open(self.results_file, "w") as f:
			json.dump(to_write, f, indent=4)

	def run_batch_queue(self, file_name: Optional[str] = None) -> None:
		"""Run inference on the dataset using batch processing.

		Args:
			file_name (Optional[str], optional): The file to store the
				requests to be processed. If `None`, a temporary file is
				used. Defaults to `None`.
		"""
		tasks = [
			{
				"prompt": instance[self.prompt_feature],
				"id": str(idx),
			}
			for idx, instance in enumerate(self.dataset)
		]

		results = self.model.generate_batch(
			[task["prompt"] for task in tasks],
			max_context_length=self.max_context_length,
			max_output_length=self.max_output_length,
			ids=[task["id"] for task in tasks],
			batch_size=self.batch_size,
			file_name=file_name,
		)

		output = []
		for result, task in zip(results, tasks, strict=True):
			output.append({"text": result, "id": task["id"]})

		with open(self.results_file, "w") as f:
			json.dump(output, f, indent=4)

	def _process_instance(self, instance: Dict[str, str | List[str]]) -> None:
		prompt = instance[self.prompt_feature]

		if self.batch_size is None:
			generation = self.model.generate(
				prompt,
				self.max_context_length,
				self.max_output_length,
				# vs_id=instance["repo"].replace("/", "_"),
			)
			prompt = [prompt]
			generation = [generation]
		else:
			generation = self.model.generate_batch(
				prompt,
				self.max_context_length,
				self.max_output_length,
			)

		for promp, gen in zip(prompt, generation, strict=True):
			to_write = {
				"prompt": promp,
				"generation": gen,
			}

			with open(self.results_file, "a") as f:
				f.write(json.dumps(to_write) + "\n")
