import json
import time
from argparse import ArgumentParser
from typing import Dict, List, TypedDict

import requests
from tqdm.auto import tqdm


class Issue(TypedDict):
	"""A GitHub issue.

	This class represents a GitHub issue with a title, body, and a list
	of comments.
	"""

	title: str
	body: str
	comments: List[str]


def _get_issues(repo: str, wait: int = 3) -> Dict[int, Issue]:
	url = f"https://api.github.com/repos/{repo}/issues"
	headers = {
		"X-GitHub-Api-Version": "2022-11-28",
		"State": "closed",
		"Accept": "application/vnd.github+json",
	}
	while True:
		try:
			response = requests.get(url, headers=headers)
			response.raise_for_status()
			break
		except (requests.RequestException, requests.HTTPError):
			time.sleep(wait)
	issues = response.json()
	return {
		issue["number"]: {
			"title": issue["title"],
			"body": issue["body"],
			"comments": [],
		}
		for issue in issues
	}


def _get_issue_comments(
	repo: str, issue_number: int, wait: int = 3
) -> List[str]:
	url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
	headers = {
		"X-GitHub-Api-Version": "2022-11-28",
		"Accept": "application/vnd.github+json",
	}
	while True:
		try:
			response = requests.get(url, headers=headers)
			response.raise_for_status()
			break
		except (requests.RequestException, requests.HTTPError):
			time.sleep(wait)
	comments = response.json()
	return [comment["body"] for comment in comments]


def _main(repos: List[str], wait: int = 3) -> Dict[int, Issue]:
	issues = {}
	for repo in repos:
		issues.update(_get_issues(repo, wait=wait))
		for issue_number in tqdm(
			issues, desc=f"Retrieving comments for repository {repo}"
		):
			issues[issue_number]["comments"] = _get_issue_comments(
				repo, issue_number, wait=wait
			)
	return issues


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--repo", type=str, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument(
		"--wait",
		type=int,
		default=3,
		help="Time to wait between retries in seconds",
	)
	args = parser.parse_args()

	issues = _main([args.repo], args.wait)
	with open(args.output, "w") as file:
		json.dump(issues, file, indent=4)
