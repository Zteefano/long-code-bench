import asyncio
import os
from typing import Dict, List, Optional, override

import anthropic
from anthropic.types.message_create_params import (
	MessageCreateParamsNonStreaming,
)
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from src.long_code_bench.models.api import APIModel

load_dotenv()


class AnthropicModel(APIModel):
	"""A class to interact with the Anthropic API for text generation.

	Args:
		model_version (str): The version of the model to use.
		api_key (Optional[str]): The API key to authenticate with the
			Anthropic API. If `None`, the API key is retrieved from the
			environment variable `ANTHROPIC_API_KEY`. By default,
			`None`.
		max_window (int): The maximum window size for the model. By
			default, `200_000`.
	"""

	def __init__(
		self,
		model_version: str,
		api_key: Optional[str] = None,
		max_window: int = 200_000,
	) -> None:
		self.model_version = model_version
		self._max_window = max_window

		api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
		self.client: anthropic.Anthropic = anthropic.Anthropic(api_key=api_key)

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
				the context to consider. Only present for compatibility,
				but not used. By default, `None`.
			max_output_length (Optional[int]): The maximum length of the
				output text. Only present for compatibility, but not
				used. By default, `None`.

		Returns:
			str: The generated text.
		"""
		messages = self.client.messages.create(
			model=self.model_version,
			max_tokens=self.max_window or 200_000,
			messages=[{"role": "user", "content": prompt}],
		)
		return messages.content[0].text  # type: ignore

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

		The main feature of this method is that it generates text
		through batch API requests, which are handled asynchronously.
		This incurs a delay in the response time, but is more efficient
		in terms of costs.

		Args:
			prompts (List[str]): The list of prompts to generate text
				from.
			max_context_length (Optional[int]): The maximum length of
				the context to consider. Only present for compatibility,
				but not used. By default, `None`.
			max_output_length (Optional[int]): The maximum length of the
				output text. Only present for compatibility, but not
				used. By default, `None`.
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
		"""
		assert ids is not None, "IDs must be provided for batch generation"

		async def _generate_batches() -> List[str]:
			tasks = []
			for i in range(0, len(prompts), batch_size):
				prompts_batch = prompts[i : i + batch_size]
				ids_batch = ids[i : i + batch_size]
				tasks.append(self._create_batch_file(prompts_batch, ids_batch))

			results = await asyncio.gather(*tasks, return_exceptions=True)
			combined_results = {}
			for result in results:
				if not isinstance(result, BaseException):
					combined_results.update(result)

			to_return = []
			for instance_id in ids:
				to_return.append(combined_results.get(instance_id, ""))
			return to_return

		results = asyncio.run(_generate_batches())

		if file_name:
			with open(file_name, "w") as f:
				for result in results:
					f.write(result + "\n")

		return results

	async def _create_batch_file(
		self, prompts_batch: List[str], ids_batch: List[str]
	) -> Dict[str, str]:
		batch_id = self.client.messages.batches.create(
			requests=[
				self._make_request(prompt, id)
				for prompt, id in zip(prompts_batch, ids_batch, strict=True)
			]
		).id

		while True:
			job = self.client.messages.batches.retrieve(batch_id)
			if job.processing_status == "ended":
				break
			await asyncio.sleep(180)

		job_results = self.client.messages.batches.results(batch_id)
		to_return = {}
		for result in job_results:
			id = result.custom_id[8:]
			curr_result = result.result
			if curr_result.type != "succeeeded":
				to_return[id] = ""
			else:
				to_return[id] = curr_result.content[0].text

		return to_return

	def _make_request(self, prompt: str, id: str) -> Request:
		return Request(
			custom_id=f"request-{id}",
			params=MessageCreateParamsNonStreaming(
				model=self.model_version,
				max_tokens=self.max_window or 200_000,
				messages=[{"role": "user", "content": prompt}],
			),
		)

	@property
	@override
	def name(self) -> str:
		"""Name of the model, used for identification."""
		return self.model_version
