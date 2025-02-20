import os
import shutil
import subprocess
import time
import types
from pathlib import Path
from typing import Dict, List, Literal, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm.auto import tqdm

load_dotenv()


class Issue(BaseModel):
	"""A GitHub issue.

	This class represents a GitHub issue with a title, body, and a list
	of comments.
	"""

	number: int
	title: str
	body: str
	comments: List[str]
	state: Literal["open", "closed"]


class GitHubRepository:
	"""Class to represent a GitHub repository.

	This class provides methods to interact with a GitHub repository,
	like cloning it locally, retrieving issues, _etc._

	Args:
		name (str): The name of the GitHub repository in the format
			`"owner/repo_name"` (_e.g._ `"yaml/pyyaml"`).
	"""

	def __init__(self, name: str) -> None:
		self.repo = name

		url = f"https://api.github.com/repos/{name}"
		while True:
			try:
				response = requests.get(url)
				if response.status_code == 404:
					raise ValueError(f"Repository {name} not found")
				response.raise_for_status()
				break
			except requests.RequestException:
				time.sleep(1)

		self._cloned_path: Optional[Path] = None

	def clone(self, path: str | Path) -> None:
		"""Clone the repository to a specified path.

		This method clones the repository to a specified path. If the
		repository is already cloned, it will update it instead of
		cloning it again.

		Args:
			path (str | Path): The path where to clone the repository.
		"""
		if self._cloned_path is None:
			self._cloned_path = Path(path) / self.short_name

		if not self._cloned_path.exists():
			subprocess.run(
				[
					"git",
					"clone",
					f"https://github.com/{self.repo}.git",
					str(self._cloned_path),
				],
				check=True,
			)
		else:
			subprocess.run(
				["git", "-C", str(self._cloned_path), "pull"], check=True
			)

	def read_file(self, file_name: str) -> Optional[str]:
		"""Retrieve the content of a file in the repository.

		This method retrieves the content of a file in the repository
		and returns it as a string, or `None` if the any error occurs
		while reading the file (_e.g._ file not found).

		Args:
			file_name (str): The name of the file to retrieve. The name
				should be relative to the root of the repository.

		Returns:
			Optional[str]: The content of the file as a string, or
				`None` if the file does not exist or an error occurs.

		Raises:
			RuntimeError: If the repository is not cloned yet.
		"""
		if self._cloned_path is None:
			raise RuntimeError(
				"Repository not cloned. Use with-statement to clone repository"
				" before accessing files."
			)

		try:
			with open(
				self._cloned_path / file_name, "r", encoding="utf-8"
			) as f:
				return f.read()
		except Exception:
			return None

	def list_files(self) -> List[str]:
		"""List all files in the repository.

		This method lists all files in the repository and returns
		them as a list of strings.

		Returns:
			list[str]: A list of strings with the names of all files in
				the repository. The names are relative to the root of
				the repository.

		Raises:
			RuntimeError: If the repository is not cloned yet.
		"""
		if self._cloned_path is None:
			raise RuntimeError(
				"Repository not cloned. Use with-statement to clone repository"
				" before accessing files."
			)

		return [
			str(file.relative_to(self._cloned_path))
			for file in self._cloned_path.rglob("*")
			if file.is_file()
		]

	def read_files(self) -> Dict[str, str]:
		"""Retrieve the content of all files in the repository.

		This method retrieves the content of all files in the repository
		and returns them as a dictionary, where the keys are the file
		names and the values are the file contents.

		Returns:
			dict[str, str]: A dictionary containing the file names as
				keys and their contents as values.

		Raises:
			RuntimeError: If the repository is not cloned yet.
		"""
		if self._cloned_path is None:
			raise RuntimeError(
				"Repository not cloned. Use with-statement to clone repository"
				" before accessing files."
			)

		return {file: self.read_file(file) for file in self.list_files()}

	def get_issues(
		self, state: Literal["open", "closed"] = "closed", wait: int = 1
	) -> Dict[int, Issue]:
		"""Retrieve the issues of the repository.

		Args:
			state (Literal["open", "closed"]): The state of the issues
				to retrieve. Can be either `"open"` or `"closed"`.
				Default is `"closed"`.
			wait (int): The number of seconds to wait between API calls.
				Default is `1`.

		Returns:
			dict[int, Issue]: A dictionary containing the issues of the
				repository. The keys are the issue numbers and the
				values are the issues themselves.
		"""
		url = f"https://api.github.com/repos/{self.repo}/issues?state={state}"
		headers = {
			"X-GitHub-Api-Version": "2022-11-28",
			"Accept": "application/vnd.github+json",
		}
		token = os.getenv("GITHUB_API_KEY")
		if token:
			headers["Authorization"] = f"Bearer {token}"

		while True:
			try:
				response = requests.get(url, headers=headers)
				response.raise_for_status()
				break
			except (requests.RequestException, requests.HTTPError):
				time.sleep(wait)
		issues = response.json()

		to_return = {
			iss["number"]: {
				"title": iss["title"],
				"body": iss["body"],
				"comments": [],
				"state": iss["state"],
			}
			for iss in issues
		}

		for issue_number in tqdm(
			to_return,
			desc=f"Retrieving issues' comments for repository {self.repo}",
		):
			to_return[issue_number]["comments"] = self._get_issue_comments(
				issue_number, wait
			)

		return to_return

	def _get_issue_comments(
		self, issue_number: int, wait: int = 1
	) -> List[str]:
		"""Retrieve the comments of a specific issue.

		Args:
			issue_number (int): The number of the issue to retrieve the
				comments from.
			wait (int): The number of seconds to wait between API calls.
				Default is `1`.

		Returns:
			list[str]: A list of strings containing the comments of the
				issue.
		"""
		url = f"https://api.github.com/repos/{self.repo}/issues/{issue_number}/comments"
		headers = {
			"X-GitHub-Api-Version": "2022-11-28",
			"Accept": "application/vnd.github+json",
		}
		token = os.getenv("GITHUB_API_KEY")
		if token:
			headers["Authorization"] = f"Bearer {token}"

		while True:
			try:
				response = requests.get(url, headers=headers)
				response.raise_for_status()
				break
			except (requests.RequestException, requests.HTTPError):
				time.sleep(wait)

		return [comm["body"] for comm in response.json()]

	def __enter__(self) -> "GitHubRepository":
		"""Enter the context manager for the cloned repository.

		This method clones the repository to a temporary directory or
		a specified one. This allows to retrieve files' content without
		incurring in too many API calls.

		Returns:
			GitHubRepository: The repository object with access to the
				cloned repository.
		"""
		tmp_dir = os.getenv("TMPDIR", "/tmp")
		path = Path(tmp_dir)

		self.clone(path)

		return self

	def __exit__(
		self,
		exc_type: Optional[type[BaseException]],
		exc_value: Optional[BaseException],
		traceback: Optional[types.TracebackType],
	) -> None:
		"""Exit the context manager and remove the cloned repository.

		This method removes the cloned repository from the temporary
		directory or the specified one.

		Args:
			exc_type (Optional[Type[BaseException]]): The type of the
				exception raised, if any.
			exc_value (Optional[BaseException]): The value of the
				exception raised, if any.
			traceback (Optional[TracebackType]): The traceback of the
				exception raised, if any.
		"""
		if self._cloned_path and self._cloned_path.exists():
			shutil.rmtree(self._cloned_path)

	@property
	def short_name(self) -> str:
		"""Return the short name of the repository."""
		return self.repo.split("/")[-1]
