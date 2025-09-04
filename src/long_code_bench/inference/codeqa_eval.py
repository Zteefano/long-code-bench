import json
import re
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from src.long_code_bench.models import Model


class LongCodeQAEvaluator:
	"""Class for running inference and evaluation on LongCodeQA.

	This class takes a model and a LongCodeQA dataset loaded from a JSON
	file and runs inference on the dataset, providing a generation for
	each instance in the dataset. It then evaluates the responses by
	parsing the final answers and comparing them with the correct
	answers.

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

	def _parse_final_answer(self, response: str) -> Optional[str]:
		"""Parse the final answer from the model response.

		Looks for patterns like 'Final Answer: A' or 'Final Answer: B'
		and extracts the letter.

		Args:
			response (str): The model's response text.

		Returns:
			Optional[str]: The extracted letter (A, B, C, or D) or None
				if not found.
		"""
		match = re.search(r"Final Answer:\s*([ABCD])", response, re.IGNORECASE)
		if match:
			return match.group(1).upper()

		match = re.search(r"\b([ABCD])\s*$", response.strip(), re.IGNORECASE)
		if match:
			return match.group(1).upper()

		return None

	def _evaluate_predictions(
		self, predictions: List[Dict]
	) -> Dict[str, float]:
		"""Evaluate the predictions and calculate accuracy.

		Args:
			predictions (List[Dict]): List of prediction dictionaries.

		Returns:
			Dict[str, float]: Dictionary containing accuracy metrics.
		"""
		correct = 0
		total = 0

		for pred in predictions:
			total += 1
			predicted_letter = self._parse_final_answer(pred["text"])

			if predicted_letter is not None:
				if predicted_letter == pred["correct_letter"]:
					correct += 1

		accuracy = correct / total if total > 0 else 0.0

		return {
			"accuracy": accuracy,
			"correct": correct,
			"total": total,
		}

	def run(self) -> None:
		"""Run inference and evaluation on the dataset."""
		predictions = []

		bar = tqdm(total=len(self.dataset), desc="Processing instances")

		if self.batch_size is None:
			for idx, instance in enumerate(self.dataset):
				prompt = instance[self.prompt_feature]
				generation = self.model.generate(
					prompt,
					self.max_context_length,
					self.max_output_length,
				)

				predictions.append(
					{
						"id": str(idx),
						"prompt": prompt,
						"response": generation,
						"correct_letter": instance["correct_letter"],
					}
				)
				bar.update(1)
		else:
			for i in range(0, len(self.dataset), self.batch_size):
				batch = self.dataset[i : i + self.batch_size]
				prompts = [instance[self.prompt_feature] for instance in batch]

				generations = self.model.generate_batch(
					prompts,
					max_context_length=self.max_context_length,
					max_output_length=self.max_output_length,
				)

				for j, (instance, generation) in enumerate(
					zip(batch, generations, strict=True)
				):
					predictions.append(
						{
							"id": str(i + j),
							"prompt": instance[self.prompt_feature],
							"response": generation,
							"correct_letter": instance["correct_letter"],
						}
					)

				bar.update(len(batch))

		bar.close()

		metrics = self._evaluate_predictions(predictions)
		output = {"predictions": predictions, "metrics": metrics}

		with open(self.results_file, "w") as f:
			json.dump(output, f, indent=2)

		print("Evaluation completed!")
		print(f"Accuracy: {metrics['accuracy']:.4f}")
		print(f"Correct: {metrics['correct']}/{metrics['total']}")

	def run_batch_queue(self, file_name: Optional[str] = None) -> None:
		"""Run inference using batch processing then evaluate.

		Args:
			file_name (Optional[str], optional): The file to store the
				requests to be processed. If `None`, a temporary file is
				used. Defaults to `None`.
		"""
		tasks = [
			{
				"prompt": instance[self.prompt_feature],
				"id": str(idx),
				"correct_letter": instance["correct_letter"],
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

		predictions = []
		for result, task in zip(results, tasks, strict=True):
			predictions.append(
				{
					"id": task["id"],
					"prompt": task["prompt"],
					"response": result,
					"correct_letter": task["correct_letter"],
				}
			)
		metrics = self._evaluate_predictions(predictions)
		output = {"predictions": predictions, "metrics": metrics}
		with open(self.results_file, "w") as f:
			json.dump(output, f, indent=2)

		print("Evaluation completed!")
		print(f"Accuracy: {metrics['accuracy']:.4f}")
		print(f"Correct: {metrics['correct']}/{metrics['total']}")
