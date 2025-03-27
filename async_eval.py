import json
import asyncio
from typing import Generator, List, Optional, Union
import os

import datasets as dts
from tqdm.auto import tqdm
from openai import OpenAI
from transformers import AutoTokenizer


def _iterate_dataset(
    dataset: Union[dts.Dataset, list, dts.DatasetDict],
    batch_size: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Yield instances (or batches) from a dataset.
    
    If the dataset is a list of strings, each string is wrapped in a dict with key "prompt".
    If it is a list of dicts, yield the dict as is.
    """
    if isinstance(dataset, list):
        # Check if elements are dicts (e.g., {"prompt": ..., "correct_letter": ...})
        if dataset and isinstance(dataset[0], dict):
            if batch_size is None:
                for instance in dataset:
                    yield instance
            else:
                len_dataset = len(dataset)
                for i in range(0, len_dataset, batch_size):
                    yield dataset[i : i + batch_size]
        else:
            # Assume list of strings.
            if batch_size is None:
                for prompt in dataset:
                    yield {"prompt": prompt}
            else:
                len_dataset = len(dataset)
                for i in range(0, len_dataset, batch_size):
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
        batch_size: Optional[int] = 2,
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
        # Instead of clearing the file, count already processed instances
        processed_count = 0
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                processed_count = sum(1 for line in f if line.strip())
        else:
            open(self.results_file, "a").close()

        total_instances = self._len_dataset()
        remaining_instances = total_instances - processed_count
        progress_bar = tqdm(total=remaining_instances, desc="Processing instances")

        dataset_iterator = _iterate_dataset(self.dataset, self.batch_size)

        if self.batch_size is None:
            # When not in batch mode, skip processed instances directly.
            for _ in range(processed_count):
                next(dataset_iterator, None)
        else:
            # In batch mode, each iterator yields a batch (a list of instances).
            batches_to_skip = processed_count // self.batch_size
            remainder = processed_count % self.batch_size

            # Skip the full batches.
            for _ in range(batches_to_skip):
                next(dataset_iterator, None)

            # If there's a remainder, skip that many items from the next batch.
            if remainder:
                batch = next(dataset_iterator, [])
                if batch:
                    batch = batch[remainder:]
                    # Process the remaining items in the batch, if any.
                    if batch:
                        prompts_list = [instance[self.prompt_feature] for instance in batch]
                        generations = await self._async_generate_batch(prompts_list)
                        for i, generation in enumerate(generations):
                            instance = batch[i]
                            result = {
                                "prompt": instance[self.prompt_feature],
                                "generation": generation,
                                "question": instance.get("question"),
                                "correct_letter": instance.get("correct_letter"),
                                "repo_text": instance.get("repo_text"),
                                "prompt_goal": instance.get("prompt_goal"),
                                "repo": instance.get("repo"),
                            }
                            with open(self.results_file, "a") as f:
                                f.write(json.dumps(result) + "\n")
                        progress_bar.update(len(batch))

        # Continue processing remaining batches.
        if self.batch_size is None:
            for instance in dataset_iterator:
                prompt = instance[self.prompt_feature]
                generation = await self._async_generate(prompt)
                result = {
                    "prompt": prompt,
                    "generation": generation,
                    "correct_letter": instance.get("correct_letter")
                }
                with open(self.results_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                progress_bar.update(1)
        else:
            for batch in dataset_iterator:
                # Check if batch is a dict with a list of prompts (list-of-strings) or a list of dicts.
                if isinstance(batch, dict) and isinstance(batch.get(self.prompt_feature), list):
                    prompts_list = batch[self.prompt_feature]
                    generations = await self._async_generate_batch(prompts_list)
                    for i, generation in enumerate(generations):
                        result = {
                            "prompt": prompts_list[i],
                            "generation": generation,
                        }
                        with open(self.results_file, "a") as f:
                            f.write(json.dumps(result) + "\n")
                    progress_bar.update(len(prompts_list))
                else:
                    prompts_list = [instance[self.prompt_feature] for instance in batch]
                    generations = await self._async_generate_batch(prompts_list)
                    for i, generation in enumerate(generations):
                        instance = batch[i]
                        result = {
                            "prompt": instance[self.prompt_feature],
                            "generation": generation,
                            "question": instance.get("question"),
                            "correct_letter": instance.get("correct_letter"),
                            "repo_text": instance.get("repo_text"),
                            "prompt_goal": instance.get("prompt_goal"),
                            "repo": instance.get("repo"),
                        }
                        with open(self.results_file, "a") as f:
                            f.write(json.dumps(result) + "\n")
                    progress_bar.update(len(prompts_list))
        progress_bar.close()


async def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Use the base_dir from the command-line argument.
    base_dir = "dataset/output_1M"
    file_path = os.path.join(base_dir, "final_dataset.json")
    result_dir = os.path.join(base_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    model_name = "AI21-Jamba-Large-1.5"
    results_file = os.path.join(result_dir, f"{model_name}.jsonl")
    
    # Load the dataset.
    with open(file_path, "r") as f:
        prompts = json.load(f)
    print("Loaded", len(prompts), "prompts")

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Large-1.5")
    max_length = 1009000

    # Filter prompts based on token length.
    kept_prompts = []
    for i, entry in enumerate(prompts):
        repo_text = entry["repo_text"]
        question = entry["question"]
        
        # Prepare the prompt text.
        prompt_text = f"""
You are a coding expert. Your task is to analyze a GitHub repository and then answer one question about it. 
Repository:
{repo_text}
Question:
{question}
Please analyze the repository text, reason through the question, and then choose among A, B, C, D answers the correct one.
Provide your final answer in the following format:
Final Answer: <LETTER>
"""
        entry["prompt"] = prompt_text
        tokenized = tokenizer(entry["prompt"])["input_ids"]
        if len(tokenized) < max_length:
            kept_prompts.append(entry)
            print(f"Kept instance {i} with token length {len(tokenized)}")
        else:
            print(f"Discarded instance {i} with token length {len(tokenized)}")
    prompts = kept_prompts
    print("Kept", len(prompts), "prompts")


    model = "/leonardo/home/userexternal/lromani0/llm_models/AI21-Jamba-1.5-Large"
    api_host = "127.0.0.1"
    api_port = 8000
    api_key = "password"

    client = OpenAI(
        base_url=f"http://{api_host}:{api_port}/v1",
        api_key=api_key,
    )

    result_dir = "results"
    os.makedirs(os.path.join(base_dir, result_dir), exist_ok=True)

    # File to save the results being the model name.
    # E.g. "/leonardo_scratch/large/userinternal/mviscia1/models/Qwen2.5-14B-Instruct-1M" becomes "Qwen2.5-14B-Instruct-1M"
    model_name = model.split("/")[-1]
    results_file = os.path.join(base_dir, result_dir, f"{model_name}.jsonl")
    print("Results will be saved to", results_file)

    evaluator = AsyncDatasetEvaluator(
        client=client,
        dataset=prompts,
        prompt_feature="prompt",  # key to extract the prompt text
        results_file=results_file,
        model=model,
    )

    await evaluator.run()
    print(f"Results written to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())

