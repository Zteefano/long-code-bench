import os
import time
from typing import List, Optional, override

from dotenv import load_dotenv
from google import genai

from src.long_code_bench.models.api import APIModel

load_dotenv()


class GeminiAIModel(APIModel):
	"""A class to interact with the Gemini API for text generation.

	Args:
		model_version (str): The version of the model to use, _e.g._,
		`"gemini-2.0-flash"`.
		api_key (Optional[str]): The API key to authenticate with
			Google `genai`. If `None`, the API key is retrieved from
			the environment variable `GOOGLE_GENAI_KEY`. By default,
			`None`.
		max_window (int): The maximum window size for the model. By
			default, `128_000`.
	"""

	def __init__(
		self,
		model_version: str,
		api_key: Optional[str] = None,
		max_window: int = 128_000,
	) -> None:
		self.model_version = model_version
		self._max_window = max_window

		api_key = api_key or os.getenv("GOOGLE_GENAI_KEY")
		self.client = genai.Client(api_key=api_key)

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
		while True:
			try:
				response = self.client.models.generate_content(
					model=self.model_version,
					contents=prompt,
				)
				return response.text
			except genai.errors.ClientError as e:
				if e.code == 429:
					time.sleep(60)
				elif e.code == 400:
					return ""
				else:
					raise e
			except Exception as e:
				raise e

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

		This method is currently not implemented for the Gemini API
		model.

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

		Raises:
			NotImplementedError: This method is not implemented for
				the Gemini API model.

		Returns:
			List[str]: The list of generated texts.
		"""
		raise NotImplementedError(
			"This method is not implemented for the Gemini API model."
		)

	@property
	@override
	def name(self) -> str:
		"""Name of the model, used for identification."""
		return self.model_version
