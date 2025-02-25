import json
import os
from argparse import ArgumentParser
from typing import Dict, List, Literal, Optional

import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm.auto import tqdm

import long_code_bench.data.repo as gitrep

load_dotenv()


class IssueToQA(BaseModel):
	"""The QA representation of a GitHub issue.

	This class first checks if the issue is simple enough that its
	resolution boils down to clarifying a concept about the codebase.
	If so, it stores a multiple choice question about the underlying
	concept, with a correct answer and three wrong answers.
	"""

	is_qa: bool
	why_is_or_is_not_qa: str
	question: Optional[str] = None
	correct_answer: Optional[str] = None
	incorrect_answers: Optional[List[str]] = None


def _issuetoqa_to_dict(issue: IssueToQA) -> Dict[str, str | List[str]]:
	if issue.is_qa:
		return {
			"is_qa": issue.is_qa,
			"question": issue.question,
			"correct_answer": issue.correct_answer,
			"incorrect_answers": issue.incorrect_answers,
		}
	else:
		return {
			"is_qa": issue.is_qa,
			"why_is_or_is_not_qa": issue.why_is_or_is_not_qa,
		}


def _issue_to_qa(issue: gitrep.Issue, client: openai.OpenAI) -> IssueToQA:
	messages = [
		{
			"role": "system",
			"content": """You are an expert software engineer and teacher. You
are given a closed GitHub issue from a public repository and the comments on
the issue. Read the issue and assess whether it is simple enough that its
resolution boils down to clarifying a concept about the codebase. If so,
reexpress it as a multiple choice question with a correct answer and three
wrong answers.

Issue whose resolution required a pull request or a code change are examples
of issues that are not simple enough for this task.

Moreover, the question should be about both the underlying concept and the
codebase, not a general question about programming, software engineering, or
the language being used.

Examples of issues that are simple enough to be expressed as multiple choice
questions:
- Clarifying the behavior of a function, method, or module.
- Understanding the purpose of a variable or class.
- Explaining the flow of control in a code snippet.
- Explaining why a certain error occurs in a code snippet due to a subtle, but
  wanted, behavior of the library being used.

Examples of issues that are not simple enough to be expressed as multiple
choice questions:
- Fixing a bug in the code.
- Implementing a new feature.
- Suggestions on what to pair with a certain library.
- Installation issues.
- Questions about the project's roadmap.
- Observations relating to software engineering practices.

Please provide reasoning to justify your decision. Be conservative, and only
consider issues as simple enough if it is clear beyond doubt that they are.""",
		},
		{
			"role": "user",
			"content": f"""Issue {issue["title"]}: {issue["body"]}\n\n
Comments:\n\n{"\n".join(issue["comments"])}""",
		},
	]
	response = client.beta.chat.completions.parse(
		model="gpt-4o", messages=messages, response_format=IssueToQA
	)
	message = response.choices[0].message
	if message.parsed:
		return message.parsed

	return IssueToQA(is_qa=False)


def _create_dataset(
	repo_issues: Dict[str, Dict[int, IssueToQA]],
	repos: Dict[str, gitrep.GitHubRepository],
	output_path: str,
	format: Literal["pre", "post", "without"] = "post",
) -> None:
	qs = []
	prompt_goal = """You are going to be provided the content of a
repository and a question about it. Provide the answer to the question by
stating ONLY the letter associated to the question."""

	for repo, issues in repo_issues.items():
		with repos[repo] as cloned_repo:
			files = cloned_repo.read_files()
		files = {k: v for k, v in files.items() if "tests" not in k}
		repo_text = "Repository:\n"
		for file, file_content in files.items():
			repo_text += (
				f"[start of {file}]\n{file_content}\n[end of {file}]\n"
			)

		for issue in issues.values():
			question = f"""Question:
{issue["question"]}

A) {issue["correct_answer"]}
B) {issue["incorrect_answers"][0]}
C) {issue["incorrect_answers"][1]}
D) {issue["incorrect_answers"][2]}"""

			if format == "pre":
				qs.append(f"{prompt_goal}\n{question}\n{repo_text}")
			elif format == "post":
				qs.append(f"{repo_text}\n{prompt_goal}\n{question}")
			elif format == "without":
				qs.append(f"""You are going to be provided with a question
about the repository {repo}. Provide the answer to the question by stating
ONLY the letter associated to the question.

{question}""")

	with open(output_path, "w") as file:
		json.dump(qs, file, indent=4)


def _main(
	repos: List[str],
	output_dir: str,
	wait: int = 1,
	format: Literal["pre", "post", "without"] = "post",
) -> Dict[str, Dict[int, gitrep.Issue | IssueToQA]]:
	if output_dir and not os.path.exists(output_dir):
		os.makedirs(output_dir)

	gitrepos = {repo: gitrep.GitHubRepository(repo) for repo in repos}

	if output_dir and os.path.exists(f"{output_dir}/github_issues.json"):
		with open(f"{output_dir}/github_issues.json", "r") as file:
			issues = json.load(file)
	else:
		issues = {repo: {} for repo in repos}

		for repo_name in repos:
			issues[repo_name].update(
				gitrepos[repo_name].get_issues(wait=wait, state="closed")
			)

		with open(f"{output_dir}/github_issues.json", "w") as file:
			json.dump(issues, file, indent=4)

	if os.path.exists(f"{output_dir}/github_issues_qa.json"):
		with open(f"{output_dir}/github_issues_qa.json", "r") as file:
			issues_qa = json.load(file)
	else:
		issues_qa = {repo: {} for repo in repos}
		api_key = os.getenv("OPENAI_API_KEY")
		client = openai.OpenAI(api_key=api_key)

		for repo_name in issues:
			for issue_number in tqdm(
				issues[repo_name], desc="Generating multiple choice questions"
			):
				to_qa = _issue_to_qa(issues[repo_name][issue_number], client)
				if to_qa.is_qa:
					issues_qa[repo_name][issue_number] = _issuetoqa_to_dict(
						to_qa
					)

		with open(f"{output_dir}/github_issues_qa.json", "w") as file:
			json.dump(issues_qa, file, indent=4)

	_create_dataset(
		issues_qa,
		gitrepos,
		f"{output_dir}/dataset_{format}.json",
		format=format,
	)

	return issues_qa


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument(
		"--repos",
		type=str,
		required=True,
		help="Path to the file containing the list of repositories",
	)
	parser.add_argument(
		"--output",
		type=str,
		required=True,
		help="Directory where to store the dataset",
	)
	parser.add_argument(
		"--wait",
		type=int,
		default=1,
		help="Time to wait between retries in seconds",
	)
	parser.add_argument(
		"--format",
		type=str,
		default="post",
		choices=["pre", "post", "without"],
		help="Format of the dataset",
	)
	args = parser.parse_args()

	repos = []
	with open(args.repos, "r") as file:
		for line in file:
			repos.append(line.strip())

	_main(repos, args.output, args.wait, args.format)
