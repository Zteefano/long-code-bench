import json
import asyncio
from typing import Generator, List, Optional, Union

import datasets as dts
from tqdm.auto import tqdm
from openai import OpenAI
from transformers import AutoTokenizer


def _iterate_dataset(
    dataset: Union[dts.Dataset, list, dts.DatasetDict],
    batch_size: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Helper function to yield instances (or batches) from a dataset.

    If the dataset is a list of strings, each string is wrapped in a dict with key "prompt".
    If the dataset is a Hugging Face Dataset, it is handled as before.
    If batch_size is provided, then in the case of a list of strings the function yields
    a dict with key "prompt" mapping to a list of prompt strings.
    """
    # If the dataset is a list (of strings), normalize it.
    if isinstance(dataset, list):
        if batch_size is None:
            for prompt in dataset:
                yield {"prompt": prompt}
        else:
            len_dataset = len(dataset)
            for i in range(0, len_dataset, batch_size):
                # Yield a dict with the key "prompt" mapping to a list of strings.
                yield {"prompt": dataset[i : i + batch_size]}
    else:
        # Assume it's a Hugging Face Dataset (or DatasetDict)
        if batch_size is None:
            for instance in dataset:
                yield dict(instance)
        else:
            len_dataset = len(dataset)
            for i in range(0, len_dataset, batch_size):
                yield dataset[i : min(i + batch_size, len_dataset)]


class AsyncDatasetEvaluator:
    """
    Asynchronous evaluator that sends prompts to an OpenAI-compatible client.

    This class expects the dataset to be either:
      - A Hugging Face dataset (or DatasetDict), or
      - A list of strings (each a prompt), which is converted into a list of dicts.
    
    In non-batch mode, each row (a dict) is processed individually.
    In batch mode, the helper yields a dict of lists (for example, {"prompt": [list of prompts]}).

    Args:
        client: An OpenAI-compatible client with a synchronous chat.completions.create method.
        dataset: Either a Hugging Face Dataset/DatasetDict or a list of prompt strings.
        prompt_feature: The key that contains the prompt (e.g., "prompt").
        results_file: File path where results are saved.
        model: The model name (or path) to use.
        batch_size: Batch size. If None, process one-by-one.
    """
    def __init__(
        self,
        client,
        dataset: Union[dts.Dataset, list, dts.DatasetDict],
        prompt_feature: str,
        results_file: str,
        model: str,
        batch_size: Optional[int] = 16,
        **kwargs,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.prompt_feature = prompt_feature
        self.results_file = results_file
        self.dataset = dataset

    def _len_dataset(self) -> int:
        if isinstance(self.dataset, list):
            return len(self.dataset)
        elif isinstance(self.dataset, dts.Dataset):
            return len(self.dataset)
        elif isinstance(self.dataset, dts.DatasetDict):
            return sum(len(self.dataset[split]) for split in self.dataset)
        else:
            raise ValueError("Dataset must be a list or a Dataset or DatasetDict.")

    def _iterate_dataset(self) -> Generator[dict, None, None]:
        return _iterate_dataset(self.dataset, self.batch_size)

    async def _async_generate(self, prompt: str) -> str:
        """
        Asynchronously send a completion request for a single prompt.
        Since the OpenAI client’s API is synchronous, we run it in a separate thread.
        """
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
        )
        return response.choices[0].message.content

    async def _async_generate_batch(self, prompts: List[str]) -> List[str]:
        tasks = [self._async_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def run(self) -> None:
        # Clear or create the results file.
        open(self.results_file, "w").close()
        total_instances = self._len_dataset()
        progress_bar = tqdm(total=total_instances, desc="Processing instances")

        if self.batch_size is None:
            for instance in _iterate_dataset(self.dataset):
                prompt = instance[self.prompt_feature]
                generation = await self._async_generate(prompt)
                result = {"prompt": prompt, "generation": generation}
                with open(self.results_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                progress_bar.update(1)
        else:
            for batch in _iterate_dataset(self.dataset, self.batch_size):
                # In the list-of-strings case, batch is a dict with key "prompt" mapping to a list.
                prompts = batch[self.prompt_feature]
                generations = await self._async_generate_batch(prompts)
                for i, generation in enumerate(generations):
                    result = {"prompt": prompts[i], "generation": generation}
                    with open(self.results_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                progress_bar.update(len(prompts))
        progress_bar.close()


async def main():
    # Load the JSON file which is assumed to contain a list of prompt strings.
    with open("/Users/lucaromani/Lavoro/Datasets/dataset_post.json", "r") as f:
        prompts = json.load(f)  # For example: ["prompt one", "prompt two", ...]

    print("Loaded", len(prompts), "prompts")

    tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-1.5-Large")

    # Remove prompts that are too long for the model and print the indices of the kept prompts.
    max_length = 128000

    # Keep only the prompts that are shorter than the maximum length.
    kept_prompts = []

    for i, prompt in enumerate(prompts):
        if len(tokenizer(prompt)["input_ids"]) < max_length:
            kept_prompts.append(prompt)
            print(f"Kept prompt {i} with length {len(tokenizer(prompt)['input_ids'])}")
    prompts = kept_prompts

    print("Kept", len(prompts), "prompts")

    # Model and API parameters.
    model = "/leonardo_scratch/large/userinternal/mviscia1/models/Llama-3.1_405B-Instruct"
    api_host = "127.0.0.1"
    api_port = 8000
    api_key = "password"

    # Create the OpenAI-compatible client.
    client = OpenAI(
        base_url=f"http://{api_host}:{api_port}/v1",
        api_key=api_key,
    )

    results_file = "output/Llama_QA.jsonl"

    # Create the evaluator.
    # Here, we pass the list of prompts (loaded from the JSON file)
    # and tell the evaluator that the prompt is stored under the key "prompt".
    evaluator = AsyncDatasetEvaluator(
        client=client,
        dataset=prompts,  # List of strings
        prompt_feature="prompt",  # Our _iterate_dataset function wraps each string in {"prompt": ...}
        results_file=results_file,
        model=model,
    )

    await evaluator.run()
    print(f"Results written to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())
