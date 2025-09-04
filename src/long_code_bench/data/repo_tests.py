import re
from dataclasses import dataclass
from typing import List, Literal, Optional

import long_code_bench.data.repo as gitrep
from long_code_bench.data.mask_functions import mask_definitions


@dataclass
class TestRunConfig:
	"""Configuration for running tests in a Docker container.

	For different repositories, the `Dockerfile`, the command to run
	tests, and the location of the source files may vary. This class
	encapsulates the configuration for a specific repository.
	"""

	dockerfile: str
	command: str
	src_dir: str


def extract_tests(
	logs: Optional[str],
	logs_file: Optional[str] = None,
	result: Literal["FAILED", "PASSED"] = "FAILED",
) -> List[str]:
	"""Extracts tests that failed or passed from the `pytest` logs.

	Args:
		logs (Optional[str]): Logs content. If `None`, the logs are read
			from the file specified in `logs_file`.
		logs_file (str): Path to the logs file. The logs are read from
			this file if `logs` is `None`.
		result (Literal["FAILED", "PASSED"]): Result to extract.

	Raises:
		ValueError: If neither `logs` nor `logs_file` is provided.

	Returns:
		List[str]: List of tests that failed or passed.
	"""
	tests = []

	if not logs:
		if not logs_file:
			raise ValueError("Either logs or logs_file must be provided.")
		with open(logs_file, "r") as f:
			logs = f.read()

	pattern = rf"\n(.*?) {result}\s+\[\s+[0-9]+%\]\n"
	matches = re.finditer(pattern, logs)
	for match in matches:
		test_name = match.group(1)
		tests.append(test_name)

	return tests


def tests_to_related_defs(
	repo: str | gitrep.GitHubRepository, defs: List[str], config: TestRunConfig
) -> List[str]:
	"""Extracts tests related to the provided definitions.

	A test is considered related to the set of definitions if it fails
	when the definitions are removed from the code, but passes when
	they are present.

	Args:
		repo (str | GitHubRepository): Repository to work with. If a
			string is provided, it is assumed to be the repository's
			name in the format `"owner/repo"`.
		defs (List[str]): List of definitions in the format
			`"file_path::def_name"`.
		config (TestRunConfig): Configuration for running tests,
			including the Dockerfile and the command to run tests.

	Returns:
		List[str]: List of tests related to the provided definitions.
	"""
	if isinstance(repo, str):
		repo = gitrep.GitHubRepository(repo)

	with repo as r:
		image = r.build_docker_image(config.dockerfile, force_rebuild=True)
		tests_base_logs = r.run_with_docker(config.command, image)
		failed_base = set(extract_tests(logs=tests_base_logs, result="FAILED"))
		print(f"Done with base tests: {len(failed_base)}")

		mask_definitions(defs)

		print("Starting masked tests")
		image = r.build_docker_image(config.dockerfile, force_rebuild=True)
		tests_masked_logs = r.run_with_docker(config.command, image)
		print(tests_masked_logs)
		failed_masked = set(
			extract_tests(logs=tests_masked_logs, result="FAILED")
		)
		print(f"Done with masked tests: {len(failed_masked)}")

		related_tests = failed_masked - failed_base

	return list(related_tests)
