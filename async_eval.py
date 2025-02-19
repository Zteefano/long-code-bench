import json
import asyncio
from typing import Generator, List, Optional, Union

import datasets as dts
from tqdm.auto import tqdm
from openai import AsyncOpenAI
from datasets import load_from_disk


def _iterate_dataset(
    dataset: dts.Dataset, batch_size: Optional[int] = None
) -> Generator[dict, None, None]:
    """
    Helper function to yield instances (or batches) from a Hugging Face Dataset.
    
    If batch_size is None, yields one row (as a dict) at a time.
    Otherwise, yields a slice of the dataset (which is a dict of lists).
    """
    if batch_size is None:
        for instance in dataset:
            # Ensure each instance is a dict.
            yield dict(instance)
    else:
        len_dataset = len(dataset)
        for i in range(0, len_dataset, batch_size):
            # Yield the slice as-is.
            yield dataset[i : min(i + batch_size, len_dataset)]


class AsyncDatasetEvaluator:
    """
    Asynchronous evaluator for Hugging Face datasets using an OpenAI-compatible async client.
    
    In non-batch mode, each row (dict) is processed individually.
    In batch mode, a slice (dict of lists) is passed to the async generation, so that each key maps to
    a list of values for that column.
    
    Args:
        client: An async OpenAI-compatible client with a `chat.completions.create` method.
        dataset (dts.Dataset | dts.DatasetDict): The dataset for inference.
        prompt_feature (str): The key in each instance (or in the batch dict) that contains the prompt.
        results_file (str): File path where results are saved.
        model (str): The model name (or path) to use.
        splits (Optional[List[str]]): For DatasetDict, only these splits will be used.
        max_context_length (Optional[int]): Maximum context length (if applicable).
        max_output_length (Optional[int]): Maximum output length (if applicable).
        batch_size (Optional[int]): Batch size. If None, process one-by-one.
    """
    def __init__(
        self,
        client,
        dataset: Union[dts.Dataset, dts.DatasetDict],
        prompt_feature: str,
        results_file: str,
        model: str,
        splits: Optional[List[str]] = None,
        max_context_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
        batch_size: Optional[int] = 16,
    ) -> None:
        self.client = client
        self.model = model
        self.max_context_length = max_context_length
        self.max_output_length = max_output_length
        self.batch_size = batch_size
        self.prompt_feature = prompt_feature
        self.results_file = results_file

        # If the dataset is a DatasetDict and splits are provided, use only those splits.
        self.dataset = dataset
        if isinstance(dataset, dts.DatasetDict) and splits is not None:
            # Here we assume `splits` is a list of keys (e.g. ["test", "validation"]).
            self.dataset = {split: dataset[split] for split in splits}

    def _len_dataset(self) -> int:
        if isinstance(self.dataset, dts.DatasetDict):
            return sum(len(self.dataset[split]) for split in self.dataset)
        elif isinstance(self.dataset, dts.Dataset):
            return len(self.dataset)
        else:
            raise ValueError("Dataset must be a Dataset or DatasetDict.")

    def _iterate_dataset(self) -> Generator[dict, None, None]:
        if isinstance(self.dataset, dts.DatasetDict):
            for split in self.dataset:
                for instance in _iterate_dataset(self.dataset[split]):
                    yield instance
        elif isinstance(self.dataset, dts.Dataset):
            for instance in _iterate_dataset(self.dataset):
                yield instance

    def _iterate_dataset_batch(self) -> Generator[dict, None, None]:
        """
        In batch mode, yield a slice of the dataset (a dict of lists) as returned by slicing.
        """
        if isinstance(self.dataset, dts.DatasetDict):
            for split in self.dataset:
                for batch in _iterate_dataset(self.dataset[split], self.batch_size):
                    yield batch
        elif isinstance(self.dataset, dts.Dataset):
            for batch in _iterate_dataset(self.dataset, self.batch_size):
                yield batch

    async def _async_generate(self, prompt: str) -> str:
        """
        Asynchronously send a completion request for a single prompt.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            # You may add additional parameters here if needed.
        )
        print(response)
        return response.choices[0].message.content

    async def _async_generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Asynchronously generate completions for a batch of prompts.
        """
        tasks = [self._async_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def _process_instance(self, instance: dict) -> None:
        """
        Process a single instance (non-batch mode): generate a completion and write the result.
        """
        prompt = instance[self.prompt_feature]
        generation = await self._async_generate(prompt)
        result = {
            "prompt": prompt,
            "generation": generation,
            "instance_id": instance.get("instance_id"),
            "num_files": instance.get("num_files"),
            "num_tokens": instance.get("num_tokens"),
        }
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    async def run(self) -> None:
        """
        Run asynchronous inference over the entire dataset.
        """
        # Clear (or create) the results file.
        open(self.results_file, "w").close()
        total_instances = self._len_dataset()
        progress_bar = tqdm(total=total_instances, desc="Processing instances")

        if self.batch_size is None:
            # Process instances one-by-one.
            for instance in self._iterate_dataset():
                await self._process_instance(instance)
                progress_bar.update(1)
        else:
            # Process in batches. In this mode, each batch is a dict of lists.
            for batch in self._iterate_dataset_batch():
                # Extract prompts from the batch directly.
                prompts = batch[self.prompt_feature]
                generations = await self._async_generate_batch(prompts)
                # Write one result per row.
                for i, generation in enumerate(generations):
                    result = {
                        "prompt": prompts[i],
                        "generation": generation,
                        "instance_id": batch["instance_id"][i] if "instance_id" in batch else None,
                        "num_files": batch["num_files"][i] if "num_files" in batch else None,
                        "num_tokens": batch["num_tokens"][i] if "num_tokens" in batch else None,
                    }
                    with open(self.results_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                progress_bar.update(len(prompts))
        progress_bar.close()


async def main():
    # Load the dataset from disk.
    dataset = load_from_disk("/Users/lucaromani/Lavoro/Datasets/swebench_ver_tuned_small")
    # Print one instance to verify its structure.
    print("First instance keys:", list(next(iter(dataset)).keys()))
    # (Expect keys such as: 'instance_id', 'num_files', 'retrieval_strategy', 'text', etc.)

    # Model and API parameters.
    model = "/leonardo/home/userexternal/lromani0/llm_models/AI21-Jamba-1.5-Large"
    api_host = "127.0.0.1"
    api_port = 8000
    api_key = "password"

    # Create the OpenAI-compatible async client.
    gen_client = AsyncOpenAI(
        base_url=f"http://{api_host}:{api_port}/v1",
        api_key=api_key,
    )

    # Specify the output file for results.
    results_file = "results.jsonl"

    # IMPORTANT:
    # Use the correct prompt field from your dataset.
    # In your dataset, the print shows keys like 'text', so we use that as the prompt.
    evaluator = AsyncDatasetEvaluator(
        client=gen_client,
        dataset=dataset,
        prompt_feature="text",  # Change this if your prompt is stored under a different key.
        results_file=results_file,
        model=model,
        batch_size=2,  # Adjust batch size as needed.
    )

    # Run the evaluator asynchronously.
    await evaluator.run()
    print(f"Results written to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
