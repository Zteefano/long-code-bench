# Benchmarking LCLMs for Coding

Repository to develop a benchmkark to test Long Context Language Models (LCLMs) coding capabilities.

## Initialization

The repository makes use of the [Pixi](https://prefix.dev/) package manager. The first step to run the code is thus to install pixi following the instructions in the link above. If on Linux or macOS, run the following command:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Once the installation is completed (restarting the terminal may be needed for it to take effect), run the following command to intall all the necessary dependencies:

```bash
pixi install
```

## API Keys

The repository needs API keys for several of its tasks. The best approach is to create an `.env` file and store the keys in there, for example:

```.env
HF_TOKEN=[Hugging Face Token for gated models]
OPENAI_API_KEY=[OpenAI API Key]
```

## Tunable SWE-Bench

This repository allows for the creation of a tunable version of the [SWE-bench](https://www.swebench.com/) benchmark, in which each problem statement is repeated with a varying number of context files. This allows for models to test their coding capabilities at different context lengths.

Before being able to create a dataset for this task, refer to [this README](src/swe_bench/swebench/inference/make_datasets/README.md) for information on how to generate a retrieval file for the dataset to process.

To create a tunable version of a SWE-Bench dataset, run the following command:

```bash
pixi r make_swebench_tuned src/long_code_bench/data/tune_swebench.py \
	--dataset [dataset_name_or_local_path] \
	--splits [splits_to_process] \
	--output_dir [directory_store_dataset] \
	--prompt_style [prompt_style] \
	--max_k [maximum_number_of_files] \
	--retrieval_type "bm25"  # Can be "random" instead
	# --retrieval_file [A retrieval file can be provided if it already exists]
```

For example, to run the processing on [SWE-Bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) run:

```bash
pixi r make_swebench_tuned \
	--dataset princeton-nlp/SWE-bench_Verified \
	--splits test \
	--output_dir [directory_store_dataset] \
	--prompt_style style-3 \
	--max_k 20 \
	--retrieval_type "bm25"
```

## Running Evaluations

For running an evaluation, run the `eval` task with Pixi (or the `eval.py` script with Python). Contrary to the `make_swebench_tuned` task, this one uses [Hydra](https://hydra.cc/) for managing runs' configurations, stored in the `conf` directory.

```bash
pixi r eval \
	dataset=swebench_verified_tuned \
	model=gpt4omini \
	output=[results_file_path]
```

## Harnessing Evaluations

Finally, after patches for a set of instances have been generated, it is possible to harness their performance with the `harness_tuned` task (or the `src/long_code_bench/inference/harness.py` script):

```bash
pixi r harness_tuned \
	--dataset [dataset_hf_identifier] \
	--predictions_path [results_file_path] \
	--max_workers [num_workers_to_use] \
	--run_id [unique_run_id] \
	--output_file [harness_results_path]
```
