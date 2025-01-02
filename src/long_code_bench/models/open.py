from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.long_code_bench.models.base import Model
import os
from typing import List, Optional, Generator, Dict
from accelerate import Accelerator

#os.environ["TRANSFORMERS_OFFLINE"] = "1"
class OpenSourceModel(Model):
	"""Class for all open-source models from the Hugging Face Hub.

	Args:
		hf_path (str): The model's path on the Hugging Face Hub.
		token (Optional[str]): The token to use for the model. By
			default, `None`.
	"""

	def __init__(
		self,
		hf_path: str,
		token: Optional[str] = None,
	) -> None:
		self.hf_path = hf_path
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		# Load tokenizer and model with optional token
		#self.tokenizer = AutoTokenizer.from_pretrained(hf_path, token=token)
		#my_path = '/leonardo/home/userexternal/asampier/IscrC_TfG/cache/cache/hub/models--ai21labs--Jamba-tiny-dev/snapshots/ed303361004ac875426a61675edecf8e9d976882'
		self.tokenizer = AutoTokenizer.from_pretrained(hf_path, #token=token, 
														#local_files_only=True,
														)
														#device_map='auto',
														#max_memory=max_memory) # auto
		
		#quantization_config = BitsAndBytesConfig(load_in_8bit=True,
        #                                 llm_int8_skip_modules=["mamba"])

		self.model = AutoModelForCausalLM.from_pretrained(
			hf_path,
			device_map='auto', # auto
			torch_dtype=torch.float16,
			#attn_implementation="flash_attention_2",
			#max_memory=max_memory,
			#token=token,
			max_memory={f"{i}": "62GB" for i in range(torch.cuda.device_count())})
			#quantization_config=quantization_config)
			#local_files_only=True,)
	
		#print(f"Model loaded successfully on rank {rank}.")
		
	

	def generate_batch(
		self,
		prompts: List[str],
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
	) -> List[str]:
		"""Generate text for a batch of prompts."""
		# Tokenize
		#print('Tokenizing...')
		inputs = self.tokenizer(
			prompts,
			max_length=max_context_length,
			truncation=max_context_length is not None,
			padding=True,
			return_tensors="pt",
		).to(self.model.device)

		# Generate
		#print('Generating...')
		outputs = self.model.generate(
			inputs["input_ids"],
			max_new_tokens=max_output_length,
		)

		#outputs = [output[0] for output in outputs]

		# Decode
		#print('Decoding...')
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
		# Encode the prompt
		inputs = self.tokenizer(
			prompt,
			max_length=max_context_length,
			truncation=max_context_length is not None,
			return_tensors="pt",
		).to(self.model.device)  # Ensures inputs are on the model's device

		# Generate output
		output = self.model.generate(
			inputs["input_ids"],
			max_new_tokens=max_output_length,
		)

		# Decode and return the generated text
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
