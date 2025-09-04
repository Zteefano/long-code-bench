# LongCodeBench

This repository contains the codebase for building the [LongCodeBench](https://arxiv.org/abs/2505.07897v2) benchmark and evaluating models on it.

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
pixi r make_swebench_tuned \
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
	dataset=[dataset_file] \
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

## CodeQA

To build the CodeQA task, run the `make_qa` task (or the `src/long_code_bench/data/codeqa.py` script):

```bash
pixi r make_qa \
	--repos [repositories_list] \
	--output [output_directory] \
	--format [prompt_format]
```

Here is the information about each argument.

* The parameter `repos` specifies a **file** with the name of a repository on each line. The name are written in the format `owner/repo_name` (an example is provided below).
* The parameter `output` specifies a **directory** where the dataset will be stored, together with intermediary files to avoid repeating GitHub API or OpenAI calls for repeated runs. In particular, if the directory already exists and it already contains some of the intermediary files (under the name `github_issues.json` and `github_issues_qa.json`), the task will skip the step that creates them.
* The paramter `format` defines the format of the prompts in which the questions are presented. The possible options are `post` (the default one), `pre`, and `without`.

Here is an example of a repositories list file:

```
yaml/pyyaml
pypa/virtualenv
jaraco/zipp
```

## Evaluating CodeQA

The same task (or script) for running an evaluation on LongSWEBench can be used to run it on LongCodeQA, as long as the dataset files (e.g., `conf/dataset/codeqa/32K.yaml`) have the `task_type` property set to `longcodeqa`.

```bash
pixi r eval \
	dataset=[dataset_file] \
	model=gpt4omini \
	output=[results_file_path]
```

Contrary to LongSWEBench, no additional step is needed and the output file will store the final accuracy of the evaluation automatically.

## SWEBench

This repository relies on the [SWEBench](https://www.swebench.com/) work by Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan1. Their repository is cloned and copied in the `src/swe_bench` directory, to which we applied changes to fit our work.

## Citations

```
@misc{rando2025longcodebenchevaluatingcodingllms,
      title={LongCodeBench: Evaluating Coding LLMs at 1M Context Windows}, 
      author={Stefano Rando and Luca Romani and Alessio Sampieri and Luca Franco and John Yang and Yuta Kyuragi and Fabio Galasso and Tatsunori Hashimoto},
      year={2025},
      eprint={2505.07897},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.07897}, 
}
```