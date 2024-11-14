import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.long_code_bench.models.base import Model


class OpenSourceModel(Model):
	"""Class for all open-source models.

	Args:
        hf_path (str): The model's path on the Hugging Face Hub.
        token (str): Optional. Hugging Face API token for accessing
            gated models.
	"""

	def __init__(self, hf_path: str, token: str = None) -> None:
		self.hf_path = hf_path
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		# Load tokenizer and model with optional token
		self.tokenizer = AutoTokenizer.from_pretrained(hf_path, token=token)
		self.model = AutoModelForCausalLM.from_pretrained(
			hf_path,
			torch_dtype=torch.float32,
			device_map="auto",  # Multi-GPU, if available
			token=token,
		)

	def generate(self, prompt: str, max_length: int = -1) -> str:
		"""Generate text given a prompt.

		Args:
            prompt (str): The prompt to generate text from.
            max_length (int): The maximum length of the generated text.
                If `-1`, the model can generate text of any length. By
                default, `-1`.

		Returns:
            str: The generated text.
		"""
		# Encode the prompt
		inputs = self.tokenizer(prompt, return_tensors="pt").to(
			self.model.device
		)  # Ensures inputs are on the model's device

		# Generate output
		output = self.model.generate(
			inputs["input_ids"],
			max_length=max_length
			if max_length > 0
			else self.model.config.max_length,
			temperature=0.7,  # Controls randomness
			top_k=50,  # Consider sampling from top-k tokens
			top_p=0.9,  # Consider sampling from top-p nucleus
			repetition_penalty=1.2,  # Penalizes repetition in generation
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
