from src.long_code_bench.models.base import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.long_code_bench.models.base import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class OpenSourceModel(Model):
    """Class for all open-source models.

    Args:
        hf_path (str): The model's path on the Hugging Face Hub.
        token (str): Optional. Hugging Face API token for accessing gated models.
    """

    def __init__(self, hf_path: str, token: str = None) -> None:
        self.hf_path = hf_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model with optional token
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token 
        )
        self.model.to(self.device)  # Move model to the appropriate device

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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate output
        output = self.model.generate(
            inputs["input_ids"],
            max_length=max_length if max_length > 0 else self.model.config.max_length,
            no_repeat_ngram_size=2,  # Optional: Avoid repeating n-grams
            do_sample=True,          # Optional: Sampling for more varied text
            top_k=50,                # Optional: Sampling from top-k tokens
            top_p=0.95               # Optional: Nucleus sampling
        )

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    @property
    def name(self) -> str:
        """Name of the model, used for identification."""
        return self.hf_path
