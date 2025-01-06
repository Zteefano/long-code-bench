from abc import ABC, abstractmethod
from typing import Optional, override

from dotenv import load_dotenv

from src.long_code_bench.models.base import Model

load_dotenv()


class APIModel(Model, ABC):
	"""An abstract class for handling API-based language models."""

	_max_window: Optional[int] = None

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
	@override
	def max_window(self) -> Optional[int]:
		"""The maximum length of text that the model can generate."""
		return self._max_window

	@property
	@abstractmethod
	def name(self) -> str:
		"""Name of the model, used for identification."""
		raise NotImplementedError
