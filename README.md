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

## Tunable SWE-Bench

This repository allows for the creation of a tunable version of the [SWE-bench](https://www.swebench.com/) benchmark, in which each problem statement is repeated with a varying number of context files. This allows for models to test their coding capabilities at different context lengths.

Before being able to create a dataset for this task, refer to [this README](src/swe_bench/swebench/inference/make_datasets/README.md) for information on how to generate a retrieval file for the dataset to process.

To create a tunable version of a SWE-Bench dataset, run the following command:

```bash
pixi r python src/long_code_bench/data/tune_swebench.py \
	--dataset [dataset_name_or_local_path] \
	--splits [splits_to_process] \
	--output_dir [directory_store_dataset] \
	--retrieval_file [retrieval_file_for_bm25] \
	--prompt_style [prompt_style] \
	--max_k [maximum_number_of_files]
	--retrieval_type "bm25"  # Can be "random" instead
	# --hfhub_dataset [path to the dataset in the Hugging Face Hub]
```

The `--hfhub_dataset` parameter is optional and, if provided, it requires defining an environment variables `"HF_TOKEN_WRITE"` with an Hugging Face token that provides writing access to the specified path.