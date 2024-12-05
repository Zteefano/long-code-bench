from typing import Optional, TypeVar

import datasets as dts
import tiktoken

T = TypeVar("T", dts.Dataset, dts.DatasetDict)


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
