from src.long_code_bench.models.base import Model
import os
import openai
import anthropic

class APIModel(Model):
    """A versatile class for handling API-based language models.
    
    This class can interface with multiple closed API-based models such as
    OpenAI's GPT and Anthropic's Claude.
    """

    def __init__(self, model_type: str, model_version: str = None,
                 api_key: str = None) -> None:
        """Initialize the API model, selecting the appropriate model type,
        version, and API key.
        
        Args:
            model_type (str): The type of model (e.g., 'openai', 'anthropic').
            model_version (str): The specific version of the model (e.g., 'gpt-3.5-turbo', 'claude-2').
            api_key (str): The API key for accessing the model's API.
        """
        self.model_type = model_type.lower()
        self.model_version = model_version or self.default_version(self.model_type)
        self.api_key = api_key or os.getenv(f"{self.model_type.upper()}_API_KEY")

        if not self.api_key:
            raise ValueError("API key is required to use this model.")
        
        if self.model_type == "openai":
            openai.api_key = self.api_key
        elif self.model_type == "anthropic":
            self.client = anthropic.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate(self, prompt: str, max_length: int = -1) -> str:
        """Generate text given a prompt by making an API call.
        
        Args:
            prompt (str): The prompt to generate text from.
            max_length (int): The maximum length of the generated text.
        
        Returns:
            str: The generated text.
        """
        if self.model_type == "openai":
            return self._generate_openai(prompt, max_length)
        elif self.model_type == "anthropic":
            return self._generate_anthropic(prompt, max_length)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _generate_openai(self, prompt: str, max_length: int) -> str:
        response = openai.Completion.create(
            model=self.model_version,
            prompt=prompt,
            max_tokens=max_length if max_length > 0 else None,
        )
        return response['choices'][0]['text']

    def _generate_anthropic(self, prompt: str, max_length: int) -> str:
        response = self.client.completions.create(
            model=self.model_version,
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=max_length if max_length > 0 else 300
        )
        return response['completion']

    @staticmethod
    def default_version(model_type: str) -> str:
        """Returns a default model version for a given model type."""
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
