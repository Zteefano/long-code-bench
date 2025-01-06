from abc import ABC, abstractmethod
from typing import List, Optional


class Model(ABC):
	"""Base class for all language models.

	A language model is a model that can generate text, given a prompt.
	Moreover, the model needs to have a name, which is used for
	identification.
	"""

	@abstractmethod
	def generate(
		self,
		prompt: str,
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
	) -> str:
		"""Generate text given a prompt.

		Args:
			prompt (str): The prompt to generate text from.
			max_context_length (Optional[int]): The maximum length of
					the context to consider. If `None`, no maximum
					length is enforced. By default, `None`.
			max_output_length (Optional[int]): The maximum length of the
					output text. If `None`, the model can generate text
					of any length. By default, `None`.

		Returns:
			str: The generated text.
		"""
		raise NotImplementedError

	@abstractmethod
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
				prompts. Relevant when the generation involves queueing
				batched jobs for API models. By default, `None`.
			file_name (Optional[str]): The file to store the generated
				results from a queued batch generation. Relevant only
				for API models that support batch generation. If `None`,
				a temporary file is used that is then deleted. By
				default, `None`.
			batch_size (int): The batch size to use when generating
				text. Only relevant for API models. By default, `10`.

		Returns:
			List[str]: The list of generated texts.
		"""
		raise NotImplementedError

	@property
	def max_window(self) -> Optional[int]:
		"""Maximum text length the model can process.

		This includes both the prompt and the generated text. It is
		`None` if there is no maximum length., which is the default
		behavior.
		"""
		return None

	@property
	@abstractmethod
	def name(self) -> str:
		"""Name of the model, used for identification."""
		raise NotImplementedError
