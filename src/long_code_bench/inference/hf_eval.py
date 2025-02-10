import json
from tqdm.asyncio import tqdm_asyncio  # Alternatively, you can use regular tqdm if you don't mind blocking on the loop.
from datasets import load_from_disk

class DatasetsEvaluator:
    def __init__(self, model, dataset_path, prompt_feature, results_file, tokenizer=None,
                 batch_size=1, system_prompt=None, max_context_length=None):
        self.model = model
        self.dataset = load_from_disk(dataset_path)
        self.prompt_feature = prompt_feature
        self.results_file = results_file
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer

        # Clear output file on startup.
        with open(self.results_file, "w") as f:
            f.write("")

    async def run(self):
        # Get the dataset split (adjust as needed)
        data = self.dataset if hasattr(self.dataset, "__len__") else self.dataset["train"]
        total = len(data)

        # You can use tqdm_asyncio or simply a normal for-loop
        for i in tqdm_asyncio(range(0, total, self.batch_size), desc="Processing instances"):
            batch = data[i : i + self.batch_size]
            batch_rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

            # Filter out examples that exceed max_context_length.
            if self.max_context_length is not None:
                valid_rows = []
                for example in batch_rows:
                    prompt = example[self.prompt_feature]
                    tokenized = self.tokenizer(prompt)
                    if len(tokenized["input_ids"]) <= self.max_context_length:
                        valid_rows.append(example)
                    else:
                        print(f"Skipping example with length {len(tokenized['input_ids'])}")
                batch_rows = valid_rows

            if not batch_rows:
                continue

            prompts = [example[self.prompt_feature] for example in batch_rows]

            # Await the asynchronous batch processing.
            generations = await self.model.process_batch(prompts, system_prompt=self.system_prompt)

            with open(self.results_file, "a") as f:
                for prompt, generation in zip(prompts, generations):
                    out = {"prompt": prompt, "generation": generation.content}
                    f.write(json.dumps(out) + "\n")
