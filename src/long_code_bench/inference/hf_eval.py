import json
from tqdm.auto import tqdm
from datasets import load_from_disk

class DatasetsEvaluator:
    def __init__(self, model, dataset_path, prompt_feature, results_file, batch_size=1, system_prompt=None):
        """
        model: an instance of APIModel
        dataset_path: path to a Hugging Face dataset on disk
        prompt_feature: the key in each dataset example that contains the prompt text
        results_file: path to the output file (e.g. JSONL)
        batch_size: number of examples to process in one batch
        system_prompt: an optional system message to include with every prompt
        """
        self.model = model
        self.dataset = load_from_disk(dataset_path)
        self.prompt_feature = prompt_feature
        self.results_file = results_file
        self.batch_size = batch_size
        self.system_prompt = system_prompt

        # Clear output file on startup.
        with open(self.results_file, "w") as f:
            f.write("")

    def run(self):
        # We assume the dataset is a Hugging Face Dataset (or DatasetDict)
        # For simplicity, we assume a single split.
        data = self.dataset if hasattr(self.dataset, "__len__") else self.dataset["train"]
        total = len(data)

        for i in tqdm(range(0, total, self.batch_size), desc="Processing instances"):
            batch = data[i : i + self.batch_size]
            batch_rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
            prompts = [example[self.prompt_feature] for example in batch_rows]
            generations = self.model.process_batch(prompts, system_prompt=self.system_prompt)

            with open(self.results_file, "a") as f:
                for prompt, generation in zip(prompts, generations):
                    out = {"prompt": prompt, "generation": generation}
                    f.write(json.dumps(out) + "\n")
