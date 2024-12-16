import os
from typing import Optional

import anthropic
import openai

from src.long_code_bench.models.base import Model


class APIModel(Model):
	"""A versatile class for handling API-based language models.

	This class can interface with multiple closed API-based models such
	as OpenAI's GPT and Anthropic's Claude.

	Args:
		model_type (str): The type of model (e.g., 'openai',
			'anthropic').
		model_version (str): The specific version of the model
			(e.g., 'gpt-3.5-turbo', 'claude-2').
		api_key (str): The API key for accessing the model's API.
	"""

	def __init__(
		self, model_type: str, model_version: str, api_key: str
	) -> None:
		self.model_type = model_type.lower()
		self.model_version = model_version or self.default_version(
			self.model_type
		)
		self.api_key = api_key or os.getenv(
			f"{self.model_type.upper()}_API_KEY"
		)

		self.client: openai.Client | anthropic.Client
		if self.model_type == "openai":
			self.client = openai.OpenAI(api_key=self.api_key)
		elif self.model_type == "anthropic":
			self.client = anthropic.Client(api_key=self.api_key)
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
			return self._generate_openai(prompt, max_output_length)
		elif self.model_type == "anthropic":
			return self._generate_anthropic(prompt, max_output_length)
		else:
			raise ValueError(f"Unsupported model type: {self.model_type}")

	def _generate_openai(self, prompt: str, max_length: Optional[int]) -> str:
		assert self.client is openai.OpenAI, "OpenAI client is not initialized"

		# Determine if the model is a chat model
		is_chat_model = (
			"gpt-3.5-turbo" in self.model_version
			or "gpt-4" in self.model_version
		)

		if is_chat_model:
			# Use ChatCompletion for chat models like gpt-4
			response = self.client.chat.completions.create(
				messages=[{"role": "user", "content": prompt}],
				model=self.model_version,
				max_length=max_length,
			)
			return response["choices"][0]["message"]["content"]
		else:
			# Use Completion for non-chat models
			response = self.client.completions.create(
				model=self.model_version,
				prompt=prompt,
				max_tokens=max_length,
			)
			return response["choices"][0]["text"]

	def _generate_anthropic(
		self, prompt: str, max_length: Optional[int]
	) -> str:
		assert (
			self.client is anthropic.Client
		), "Anthropic client is not initialized"
		response = self.client.completions.create(
			model=self.model_version,
			prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
			max_tokens_to_sample=max_length if max_length else 300,
		)
		return response["completion"]

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

	@property
	def name(self) -> str:
		"""Name of the model, used for identification."""
		return f"{self.model_type.capitalize()} Model ({self.model_version})"
