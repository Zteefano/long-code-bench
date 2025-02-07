import json
from tqdm.auto import tqdm
from datasets import load_from_disk

class DatasetsEvaluator:
    def __init__(self, model, dataset_path, prompt_feature, results_file, tokenizer = None,
                 batch_size=1, system_prompt=None, max_context_length=None):
        """
        model: an instance of APIModel, assumed to have a `tokenizer` attribute (or you can pass a tokenizer separately)
        dataset_path: path to a Hugging Face dataset on disk
        prompt_feature: the key in each dataset example that contains the prompt text
        results_file: path to the output file (e.g. JSONL)
        batch_size: number of examples to process in one batch
        system_prompt: an optional system message to include with every prompt
        max_context_length: maximum number of tokens that the model can support; examples longer than this will be filtered out.
        """
        self.model = model
        self.dataset = load_from_disk(dataset_path)
        self.prompt_feature = prompt_feature
        self.results_file = results_file
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length  # e.g., 105712

        self.tokenizer = tokenizer

        # Clear output file on startup.
        with open(self.results_file, "w") as f:
            f.write("")

    def run(self):
        # Assume the dataset is a Hugging Face Dataset (or DatasetDict) with a single split.
        data = self.dataset if hasattr(self.dataset, "__len__") else self.dataset["train"]
        total = len(data)

        for i in tqdm(range(0, total, self.batch_size), desc="Processing instances"):
            batch = data[i : i + self.batch_size]
            # Convert columnar batch into a list of row dictionaries.
            batch_rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

            # Filter out examples that exceed the maximum context length.
            if self.max_context_length is not None:
                valid_rows = []
                for example in batch_rows:
                    prompt = example[self.prompt_feature]
                    # Tokenize the prompt.
                    tokenized = self.tokenizer(prompt)
                    # Assume tokenized is a dict with an 'input_ids' key.
                    if len(tokenized["input_ids"]) <= self.max_context_length:
                        valid_rows.append(example)
                    else:
                        print(f"Skipping example with length {len(tokenized['input_ids'])}")
                batch_rows = valid_rows

            # If after filtering there are no examples, skip this batch.
            if not batch_rows:
                continue

            # Extract prompts from the filtered rows.
            prompts = [example[self.prompt_feature] for example in batch_rows]
            generations = self.model.process_batch(prompts, system_prompt=self.system_prompt)

            with open(self.results_file, "a") as f:
                for prompt, generation in zip(prompts, generations):
                    out = {"prompt": prompt, "generation": generation.content}
                    print(type(out))
                    print()
                    f.write(json.dumps(out) + "\n")
