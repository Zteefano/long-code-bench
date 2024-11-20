from abc import ABC, abstractmethod
from typing import Optional


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

	@property
	@abstractmethod
	def name(self) -> str:
		"""Name of the model, used for identification."""
		raise NotImplementedError
