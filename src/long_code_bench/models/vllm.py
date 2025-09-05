import ast
from typing import Any, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.long_code_bench.models.base import Model


def _convert_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


class OpenSourceVLLMModel(Model):
    """Class for open-source models using vLLM for inference.

    Args:
        hf_path (str): The model's path on the Hugging Face Hub.
        token (Optional[str]): Not used by vLLM.
        offline (bool): Offline flag; not used by vLLM.
        **kwargs: Additional keyword arguments for LLM initialization (e.g., tensor_parallel_size,
            max_model_len, enable_chunked_prefill, etc.).
    """
    def __init__(
        self,
        hf_path: str,
        token: Optional[str] = None,
        offline: bool = False,
        **kwargs: str,
    ) -> None:
        self.hf_path = hf_path
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
        # Convert additional keyword arguments as needed.
        llm_kwargs = {k: _convert_value(v) for k, v in kwargs.items()}
        # Create the vLLM instance; additional parameters can be passed via kwargs.
        self.llm = LLM(model=hf_path,
        tensor_parallel_size = 2,
        max_model_len=50000,
        gpu_memory_utilization = 0.6,
        **llm_kwargs)

    def generate_batch(
        self,
        prompts: List[str],
        max_context_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
        ids: Optional[List[str]] = None,
        file_name: Optional[str] = None,
        batch_size: int = 10,
    ) -> List[str]:
        """Generate text for a batch of prompts using vLLM.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            max_context_length (Optional[int]): Not used.
            max_output_length (Optional[int]): Maximum tokens to generate.
            ids (Optional[List[str]]): Not used.
            file_name (Optional[str]): Not used.
            batch_size (int): Batch size parameter; vLLM manages batching internally.

        Returns:
            List[str]: The generated texts.
        """
        sampling_params = SamplingParams(temperature=0.7,
                                         top_p=0.8,
                                         repetition_penalty=1.05)
        # Directly pass prompts to vLLM (without additional chat formatting)
        outputs = self.llm.generate(prompts, sampling_params)
        # Each output contains a prompt and a list of generated responses.
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts

    def generate(
        self,
        prompt: str,
        max_context_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
    ) -> str:
        """Generate text for a single prompt using vLLM.

        Args:
            prompt (str): The prompt text.
            max_context_length (Optional[int]): Not used.
            max_output_length (Optional[int]): Maximum tokens to generate.

        Returns:
            str: The generated text.
        """
        sampling_params = SamplingParams(temperature=0.7,
                                         top_p=0.8,
                                         repetition_penalty=1.05)
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    @property
    def name(self) -> str:
        """Name of the model, used for identification."""
        return self.hf_path
