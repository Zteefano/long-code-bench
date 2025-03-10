import os
from typing import Optional, TypeVar

import datasets as dts
import tiktoken

from long_code_bench.data.repo import GitHubRepository

T = TypeVar("T", dts.Dataset, dts.DatasetDict)


def count_repo_tokens(repo: str | GitHubRepository) -> int:
	"""Count the number of tokens in a repository.

	Args:
		repo (str | GitHubRepository): The repository to count the
			tokens in. Can be either a `str` or a `GitHubRepository`. If
			a `str`, it should be in the format `"owner/repo"`.

	Returns:
		int: The number of tokens in the repository, using the GPT-4o
			tokenizer.
	"""
	if isinstance(repo, str):
		repo = GitHubRepository(repo)

	enc = tiktoken.encoding_for_model("gpt-4o")
	with repo as r:
		files = r.read_files(
			filter=lambda x: "tests" not in x.split(os.sep)
			and ".git" not in x.split(os.sep)
		)

	tokens = 0
	for file, contents in files.items():
		try:
			tokens += len(enc.encode(contents))
		except Exception as e:
			print(f"Error encoding {file}: {e}")
			continue
	return tokens


def count_tokens(
	dataset: T,
	num_workers: int = 1,
	upload_to_hub: Optional[str] = None,
	hf_token: Optional[str] = None,
) -> T:
	"""Add a column to the dataset with context length in tokens.

	Args:
		dataset (T): The dataset to process. Can be either a `Dataset`
			or a `DatasetDict`.
		num_workers (int, optional): The number of workers to use.
			Defaults to `1`.
		upload_to_hub (Optional[str], optional): The name of the dataset
			to upload to the hub. Defaults to `None`. If `None`, the
			dataset is not uploaded.
		hf_token (Optional[str], optional): The Hugging Face token to
			upload the dataset. Defaults to `None`.

	Returns:
		T: The dataset with the new column and same type as the input.
	"""
	if upload_to_hub is not None:
		assert (
			hf_token is not None
		), "hf_token must be provided if upload_to_hub is specified."
	enc = tiktoken.encoding_for_model("gpt-4o")
	dataset = dataset.map(
		lambda instance: {"num_tokens": len(enc.encode(instance["text"]))},
		num_proc=num_workers,
	)
	if upload_to_hub:
		dataset.push_to_hub(repo_id=upload_to_hub, token=hf_token)
	return dataset
