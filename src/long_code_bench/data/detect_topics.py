import base64
import json
import logging
import pathlib
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Optional

import datasets as dts
import editdistance
import openai
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
	level=logging.NOTSET,
	format="[%(levelname)s] [%(asctime)s] [%(message)s]",
	datefmt="%Y-%m-%d %H:%M:%S",
)


class RepoTopics(BaseModel):
	"""A class to store the topics of a repository."""

	class Topic(BaseModel):
		"""A class to store a topic and its justification."""

		name: str
		justification: str

	topics: List[Topic]


def _filter_topics_by_edit_distance(
	topics: List[str], possible_topics: List[str], threshold: int = 3
) -> List[str]:
	filtered_topics = []
	for topic in topics:
		closest_topic = min(
			possible_topics, key=lambda x: editdistance.eval(x, topic)
		)
		if editdistance.eval(closest_topic, topic) <= threshold:
			filtered_topics.append(closest_topic)
	return filtered_topics


def _get_repo_info(repo_name: str) -> dict:
	logging.info(f"Getting repository info: {repo_name}")

	repo_api_url = f"https://api.github.com/repos/{repo_name}"
	response = requests.get(repo_api_url)
	response.raise_for_status()
	data = response.json()
	return {
		"name": data["name"],
		"description": data["description"],
	}


def _get_readme(repo_name: str) -> str:
	logging.info(f"Getting repository README: {repo_name}")

	repo_api_url = f"https://api.github.com/repos/{repo_name}/readme"
	response = requests.get(repo_api_url)
	response.raise_for_status()
	data = response.json()
	return base64.b64decode(data["content"]).decode("utf-8")


def _get_cs_topics(
	description: str,
	readme: str,
	possible_topics: List[str],
	client: openai.Client,
	model: str = "gpt-4o-mini",
) -> List[str]:
	logging.info("Getting CS topics")

	prompt = f"""Given the following repository description and README, select
the CS topics covered from the provided list.\n\n
A topic is a broad subject or theme that is covered in the repository. For
example, 'Machine Learning' and 'Game Development' are topics, while 'Linear
Regression' and 'Unity Engine' are not.\n\n
Do not select topics that are not part of the provided list and include a solid
justification for every choice.\n\n
Description:\n
{description}\n\n
README:\n
{readme}\n\n
Possible topics:\n
{'\n---\n'.join(possible_topics)}\n"""

	response = client.beta.chat.completions.parse(
		model=model,
		messages=[{"role": "user", "content": prompt}],
		response_format=RepoTopics,
	)
	parsed = response.choices[0].message.parsed
	if parsed is None:
		return []

	return _filter_topics_by_edit_distance(
		[topic.name for topic in parsed.topics], possible_topics
	)


def detect_swebench_topics(
	dataset: str,
	splits: List[str],
	output_file: str,
	model: str = "gpt-4o-mini",
	possible_topics: Optional[List[str]] = None,
) -> None:
	"""Detect the topics of the repositories in a SWE-Bench dataset.

	This function uses GPT-4 to detect the topics of the repositories in
	a SWE-Bench dataset, based on the repository description and README.

	Args:
		dataset (str): The name of the dataset from the Hugging Face Hub
			or the path to the dataset on disk.
		splits (List[str]): The splits to use from the dataset.
		output_file (str): The path to the output file.
		model (str, optional): The model to use. Defaults to
			`"gpt-4o-mini"`.
		possible_topics (Optional[List[str]], optional): The list of
			possible topics. If `None`, a default list is used. Defaults
			to `None`.

	Raises:
		ValueError: If the dataset is not a `DatasetDict`.
	"""
	if possible_topics is None:
		possible_topics = [
			"Artificial Intelligence",
			"Asynchronous Programming",
			"Machine Learning",
			"Cybersecurity",
			"Computer Networks",
			"Operating Systems",
			"Databases",
			"Data Serialization",
			"Distributed Systems",
			"Cloud Computing",
			"Web Development",
			"Mobile Development",
			"Game Development",
			"Embedded Systems",
			"Information Retrieval",
			"IoT (Internet of Things)",
			"Blockchain",
			"DevOps",
			"Human-Computer Interaction",
			"Computer Graphics",
			"Virtual and Augmented Reality",
			"Robotics",
			"Natural Language Processing",
			"Operating Systems",
			"Computer Vision",
			"Algorithms and Data Structures",
			"Concurrent Programming",
			"Programming Languages",
			"Quantum Computing",
			"Theory of Computation",
			"High-Performance Computing",
			"Bioinformatics",
			"Simulation and Modeling",
			"Signal Processing",
			"Statistics",
			"Real-Time Systems",
		]

	if pathlib.Path(dataset).exists():
		data = dts.load_from_disk(dataset)
	else:
		data = dts.load_dataset(dataset)

	if not isinstance(data, dts.DatasetDict):
		raise ValueError("The dataset must be a DatasetDict.")

	client = openai.Client()

	topics_dict = {}
	repo_count = defaultdict(lambda: 0)

	for split in splits:
		logging.info(f"Processing split: {split}")

		for element in data[split]:
			repo_name = element["repo"]  # type: ignore
			if repo_name in repo_count.keys():
				logging.info(
					f"Skipping already processed repository: {repo_name}"
				)
				repo_count[repo_name] += 1
				continue

			logging.info(f"Processing repository: {repo_name}")
			repo_info = _get_repo_info(repo_name)
			readme = _get_readme(repo_name)
			topics = _get_cs_topics(
				repo_info["description"],
				readme,
				possible_topics,
				client,
				model=model,
			)
			topics_dict[repo_name] = topics
			repo_count[repo_name] += 1

	topics_occurrences = dict.fromkeys(possible_topics, 0)
	for repo, repo_topics in topics_dict.items():
		for topic in repo_topics:
			topics_occurrences[topic] += repo_count[repo]

	logging.info(f"Saving output to: {output_file}")
	with open(output_file, "w") as f:
		json.dump(
			{
				"topics_per_repo": topics_dict,
				"repo_occurences": repo_count,
				"topics_occurences": topics_occurrences,
			},
			f,
			indent=4,
		)


if __name__ == "__main__":
	parser = ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dataset",
		type=str,
		required=True,
		help="Dataset's name from the Hugging Face Hub or path to a directory"
		+ " where the dataset is stored.",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=["dev", "test"],
		help="Splits to use from the dataset.",
	)
	parser.add_argument(
		"--output_file",
		type=str,
		required=True,
		help="Path to the output file.",
	)
	parser.add_argument(
		"--possible_topics",
		nargs="*",
		default=None,
		help="List of possible topics.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="gpt-4o-mini",
		help="The model to use.",
	)
	args = parser.parse_args()

	detect_swebench_topics(
		dataset=args.dataset,
		splits=args.splits,
		output_file=args.output_file,
		possible_topics=args.possible_topics,
		model=args.model,
	)
