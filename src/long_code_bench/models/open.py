import ast
from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from src.long_code_bench.models.base import Model


def _convert_value(value: str) -> Any:  # noqa: ANN401
	try:
		return ast.literal_eval(value)
	except (ValueError, SyntaxError):
		return value


class OpenSourceModel(Model):
	"""Class for all open-source models from the Hugging Face Hub.

	Args:
		hf_path (str): The model's path on the Hugging Face Hub.
		token (Optional[str]): The token to use for the model. By
			default, `None`.
		offline (bool): Whether to load the model and the tokenizer from
			local files only. By default, `False`.
		**kwargs: Additional keyword arguments to pass to the model.
			These argument can provide additional configuration like
			`attn_implementation`, or `torch_dtype`.
	"""

	def __init__(
		self,
		hf_path: str,
		token: Optional[str] = None,
		offline: bool = False,
		**kwargs: str,
	) -> None:
		self.hf_path = hf_path
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.tokenizer = AutoTokenizer.from_pretrained(
			hf_path, token=None, local_files_only=offline, device_map="auto"
		)

		model_kwargs = {
			k: _convert_value(v)
			for k, v in kwargs.items()
			if k in PretrainedConfig.__init__.__code__.co_varnames
		}
		self.model = AutoModelForCausalLM.from_pretrained(
			hf_path,
			device_map="auto",
			token=token,
			local_files_only=offline,
			**model_kwargs,
		)

	def generate_batch(
		self,
		prompts: List[str],
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
		ids: Optional[List[str]] = None,
		file_name: Optional[str] = None,
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
			ids (Optional[List[str]): The list of IDs associated to the
				prompts. Irrelevant for open source models. By default,
				`None`.
			file_name (Optional[str]): The file to store the generated
				results from a queued batch generation. Irrelevant for
				open source models. By default, `None`.

		Returns:
			List[str]: The list of generated texts.
		"""
		inputs = self.tokenizer(
			prompts,
			max_length=max_context_length,
			truncation=max_context_length is not None,
			padding=True,
			return_tensors="pt",
		).to(self.model.device)

		outputs = self.model.generate(
			inputs["input_ids"],
			max_new_tokens=max_output_length,
		)

		generated_texts = self.tokenizer.batch_decode(
			outputs,
			skip_special_tokens=True,
			clean_up_tokenization_spaces=True,
		)
		return generated_texts

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
				the context to consider. If `None`, no maximum length is
				enforced. By default, `None`.
			max_output_length (Optional[int]): The maximum length of the
				output text. If `None`, the model can generate text of
				any length. By default, `None`.

		Returns:
			str: The generated text.
		"""
		inputs = self.tokenizer(
			prompt,
			max_length=max_context_length,
			truncation=max_context_length is not None,
			return_tensors="pt",
		).to(self.model.device)

		output = self.model.generate(
			inputs["input_ids"],
			max_new_tokens=max_output_length,
		)

		generated_text = self.tokenizer.decode(
			output[0],
			skip_special_tokens=True,
			clean_up_tokenization_spaces=True,
		)

		return generated_text

	@property
	def name(self) -> str:
		"""Name of the model, used for identification."""
		return self.hf_path
