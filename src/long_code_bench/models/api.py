import asyncio
import json
import os
import time
from typing import List, Optional, override

import anthropic
import openai
from dotenv import load_dotenv

from src.long_code_bench.models.base import Model

load_dotenv()


class APIModel(Model):
	"""A versatile class for handling API-based language models.

	This class can interface with multiple closed API-based models such
	as OpenAI's GPT and Anthropic's Claude.

	Args:
		model_type (str): The type of model (e.g., 'openai',
			'anthropic').
		model_version (str): The specific version of the model
			(e.g., 'gpt-3.5-turbo', 'claude-2').
		api_key (Optional[str]): The API key to use for the model. If
			`None`, the API key is read from the environment variable
			`{MODEL_TYPE}_API_KEY`. By default, `None`.
	"""

	def __init__(
		self,
		model_type: str,
		model_version: str,
		api_key: Optional[str] = None,
	) -> None:
		self.model_type = model_type.lower()
		self.model_version = model_version or self.default_version(
			self.model_type
		)
		self.api_key = api_key or os.getenv(
			f"{self.model_type.upper()}_API_KEY"
		)

		self.client: openai.Client | anthropic.Client
		self._max_window: Optional[int] = None
		if self.model_type == "openai":
			self.client = openai.OpenAI(api_key=self.api_key)
			self._max_window = 128_000
		elif self.model_type == "anthropic":
			self.client = anthropic.Client(api_key=self.api_key)
			self._max_window = 200_000
		else:
			raise ValueError(f"Unsupported model type: {self.model_type}")

	def generate(
		self,
		prompt: str,
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
	) -> str:
		"""Generate text given a prompt by making an API call.

		Args:
			prompt (str): The prompt to generate text from.
			max_context_length (Optional[int]): The maximum length of
				the context to consider. If `None`, no maximum length is
				enforced. By default, `None`.
			max_output_length (Optional[int]): The maximum length of the
				output text. If `None`, the model can generate text of
				any length. By default, `None`.

		Returns:
			str: The generated text.

		Raises:
			ValueError: If the model type is not supported.
		"""
		if self.model_type == "openai":
			return self._generate_openai(prompt, self.max_window)
		elif self.model_type == "anthropic":
			return self._generate_anthropic(prompt, self.max_window)
		else:
			raise ValueError(f"Unsupported model type: {self.model_type}")

	def generate_batch(
		self,
		prompts: List[str],
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
		ids: Optional[List[str]] = None,
		file_name: Optional[str] = None,
		batch_size: int = 10,
	) -> List[str]:
		"""Generate text for a batch of prompts.

		Args:
			prompts (List[str]): The list of prompts to generate text
				from.
			max_context_length (Optional[int]): The maximum length of
				the context to consider. If `None`, no maximum length is
				enforced. By default, `None`.
			max_output_length (Optional[int]): The maximum length of the
				output text. If `None`, the model can generate text of
				any length. By default, `None`.
			ids (Optional[List[str]]): The list of IDs associated to the
				prompts. Necessary to keep track of the generated text
				associated with each prompt. By default, `None`.
			file_name (Optional[str]): The file to store the generated
				results from a queued batch generation. If `None`, a
				temporary file is used that is then deleted. By default,
				`None`.
			batch_size (int): The size of the batch, _i.e._, the number
				of prompts to generate text for in each batch file. By
				default, `10`.

		Returns:
			List[str]: The list of generated texts.

		Raises:
			ValueError: If the model type is not supported.
		"""
		assert ids is not None, "IDs must be provided for batch generation"

		if self.model_type == "openai":
			return self._generate_batch_openai(
				prompts,
				ids,
				max_output_length,
				file_name=file_name,
				batch_size=batch_size,
			)
		else:
			raise ValueError(
				"Batch generation is not supported for this model"
			)

	def _generate_openai(self, prompt: str, max_length: Optional[int]) -> str:
		assert (
			type(self.client) is openai.OpenAI
		), "OpenAI client is not initialized"

		is_chat_model = (
			"gpt-3.5-turbo" in self.model_version
			or "gpt-4" in self.model_version
		)

		if is_chat_model:
			response = self.client.chat.completions.create(
				messages=[{"role": "user", "content": prompt}],
				model=self.model_version,
				max_tokens=max_length,
			)
			return response.choices[0].message.content or ""
		else:
			response = self.client.completions.create(
				model=self.model_version,
				prompt=prompt,
				max_tokens=max_length,
			)
			return response.choices[0].text

	def _generate_anthropic(
		self, prompt: str, max_length: Optional[int]
	) -> str:
		assert (
			type(self.client) is anthropic.Client
		), "Anthropic client is not initialized"
		response = self.client.completions.create(
			model=self.model_version,
			prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
			max_tokens_to_sample=max_length if max_length else 300,
		)
		return response.completion

	def _generate_batch_openai(
		self,
		prompts: List[str],
		ids: List[str],
		max_length: Optional[int],
		file_name: Optional[str] = None,
		batch_size: int = 10,
	) -> List[str]:
		assert (
			type(self.client) is openai.OpenAI
		), "OpenAI client is not initialized"
		client: openai.OpenAI = self.client

		async def _create_batch_file(prompts_batch, ids_batch):  # noqa: ANN001, ANN202
			tmp_dir = os.getenv("TMPDIR", "/tmp")
			temp_file_name = os.path.join(
				tmp_dir, f"batch_file_{int(time.time())}_{os.getpid()}.jsonl"
			)

			with open(temp_file_name, "w") as f:
				for prompt, instance_id in zip(
					prompts_batch, ids_batch, strict=True
				):
					task = {
						"custom_id": f"task-{instance_id}",
						"method": "POST",
						"url": "/v1/chat/completions",
						"body": {
							"model": self.model_version,
							"temperature": 0.1,
							"messages": [
								{"role": "user", "content": prompt},
							],
						},
					}
					f.write(json.dumps(task) + "\n")

			batch_file = client.files.create(
				file=open(temp_file_name, "rb"), purpose="batch"
			)
			batch_job = client.batches.create(
				input_file_id=batch_file.id,
				endpoint="/v1/chat/completions",
				completion_window="24h",
			)

			while True:
				job_status = client.batches.retrieve(batch_job.id)
				if (
					job_status.status == "completed"
					and job_status.output_file_id
				) or job_status.status == "failed":
					break
				await asyncio.sleep(180)

			result_file_id = batch_job.output_file_id
			if not result_file_id:
				raise ValueError("Batch job failed to generate output")
			result = client.files.content(result_file_id).content

			result_file_name = os.path.join(
				tmp_dir, f"batch_job_results_{os.getpid()}.jsonl"
			)
			with open(result_file_name, "wb") as file:
				file.write(result)

			results = []
			with open(result_file_name, "r") as file:
				for line in file:
					json_object = json.loads(line.strip())
					results.append(
						json_object["response"]["body"]["choices"][0][
							"message"
						]["content"]
					)

			os.remove(temp_file_name)

			return results

		async def _generate_batches():  # noqa: ANN202
			tasks = []
			for i in range(0, len(prompts), batch_size):
				prompts_batch = prompts[i : i + batch_size]
				ids_batch = ids[i : i + batch_size]
				tasks.append(_create_batch_file(prompts_batch, ids_batch))
			results = await asyncio.gather(*tasks)
			return [item for sublist in results for item in sublist]

		results = asyncio.run(_generate_batches())

		if file_name:
			with open(file_name, "w") as f:
				for result in results:
					f.write(result + "\n")

		return results

	@staticmethod
	def default_version(model_type: str) -> str:
		"""Returns a default model version for a given model type.

		Args:
			model_type (str): The type of model (e.g., 'openai',
				'anthropic').

		Returns:
			str: The default model version.

		Raises:
			ValueError: If the model type is not supported.
		"""
		if model_type == "openai":
			return "gpt-3.5-turbo"
		elif model_type == "anthropic":
			return "claude-2"
		else:
			raise ValueError(f"Unsupported model type: {model_type}")

	@staticmethod
	def from_model_name(model_name: str) -> "APIModel":
		"""Create an APIModel instance from a model name.

		Args:
			model_name (str): The model name. Currently supported ones
				are 'gpt-4o', 'gpt-4o-mini', 'claude-3.5'.

		Returns:
			APIModel: The APIModel instance.

		Raises:
			ValueError: If the model name is not supported.
		"""
		if model_name in ["gpt-4o", "gpt-4o-mini"]:
			return APIModel("openai", model_name)
		elif model_name == "claude-3.5":
			return APIModel("anthropic", "claude-3.5")
		else:
			raise ValueError(f"Unsupported model name: {model_name}")

	@property
	@override
	def max_window(self) -> Optional[int]:
		"""The maximum length of text that the model can generate."""
		return self._max_window

	@property
	def name(self) -> str:
		"""Name of the model, used for identification."""
		return f"{self.model_type.capitalize()} Model ({self.model_version})"
