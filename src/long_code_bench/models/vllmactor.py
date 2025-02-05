import ray
from vllm import LLM
from transformers import AutoTokenizer
from typing import List, Optional


@ray.remote
class VLLMActor:
    def __init__(
        self,
        model_path: str,
        api_key: Optional[str] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize a VLLMActor with vLLM for distributed inference.

        Args:
            model_path (str): Path to the model directory or model name.
            api_key (Optional[str]): API key (not needed for local models).
            tensor_parallel_size (int): Number of GPUs for tensor parallelism.
            pipeline_parallel_size (int): Number of GPUs for pipeline parallelism.
            cache_dir (Optional[str]): Directory for local model cache.
        """
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=True,  # Ensure it only uses the local cache
        )
        
        # Initialize the LLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            api_key=None,  # Ensure no API key is used for local models
            cache_dir=cache_dir,
        )
        print(f"[INFO]: Model and tokenizer loaded from cache at '{cache_dir}'.")

    def generate_batch(
        self,
        prompts: List[str],
        max_context_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
    ) -> List[str]:
        """
        Generate text for a batch of prompts.

        Args:
            prompts (List[str]): List of input prompts.
            max_context_length (Optional[int]): Maximum input token length (context length).
            max_output_length (Optional[int]): Maximum number of tokens to generate.

        Returns:
            List[str]: List of generated texts.
        """
        # Preprocess inputs using the tokenizer
        tokenized_inputs = [
            self.tokenizer(prompt, max_length=max_context_length, truncation=True)
            for prompt in prompts
        ]

        # Generate outputs using the LLM
        results = self.llm.generate(
            prompts=prompts,
            max_tokens=max_output_length,
        )

        # Decode outputs using the tokenizer
        generated_texts = [result.outputs[0].text for result in results]
        return generated_texts
