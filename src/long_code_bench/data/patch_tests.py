import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import docker

from swe_bench.swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swe_bench.swebench.harness.docker_build import (
	build_container,
	build_env_images,
	setup_logger,
)
from swe_bench.swebench.harness.docker_utils import (
	cleanup_container,
	copy_to_container,
	exec_run_with_timeout,
)
from swe_bench.swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swe_bench.swebench.harness.test_spec import TestSpec, make_test_spec
from swe_bench.swebench.harness.utils import load_swebench_dataset


def _process_instance(
	instance: TestSpec,
	test_patch: str,
	gold_patch: str,
	client: docker.DockerClient,
) -> None:
	log_file = Path(f"/tmp/{instance.instance_id}.log")
	logger = setup_logger(
		instance.instance_id,
		log_file=log_file,
	)

	container = build_container(
		instance,
		client,
		f"test-{instance.instance_id}",
		logger=logger,
		nocache=True,
		force_rebuild=True,
	)
	container.start()

	with open("/tmp/test_patch.diff", "w") as f:
		f.write(test_patch)
	copy_to_container(
		container,
		Path("/tmp/test_patch.diff"),
		Path("/tmp/test_patch.diff"),
	)

	_val = container.exec_run(
		"git apply --allow-empty -v /tmp/test_patch.diff",
		workdir="/testbed",
		user="root",
	)
	if _val.exit_code != 0:
		_val = container.exec_run(
			"patch --batch --fuzz=5 -p1 -i /tmp/test_patch.diff",
			workdir="/testbed",
			user="root",
		)

	_git_diff_output_before = (
		container.exec_run("git diff", workdir="/testbed")
		.output.decode("utf-8")
		.strip()
	)

	with open("/tmp/eval.sh", "w") as f:
		f.write(
			"\n".join(
				[
					"#!/bin/bash",
					"set -uxo pipefail",
				]
				+ instance.eval_script_list[:3]
				+ [
					MAP_REPO_VERSION_TO_SPECS[instance.repo][instance.version][
						"test_cmd"
					]
				]
			)
		)
	copy_to_container(container, Path("/tmp/eval.sh"), Path("/eval.sh"))

	test_output, _, _ = exec_run_with_timeout(
		container, "/bin/bash /eval.sh", timeout=180
	)
	print(test_output)

	pre_gold_tests = MAP_REPO_TO_PARSER[instance.repo](test_output)

	with open("/tmp/gold_patch.diff", "w") as f:
		f.write(gold_patch)
	copy_to_container(
		container,
		Path("/tmp/gold_patch.diff"),
		Path("/tmp/gold_patch.diff"),
	)

	_val = container.exec_run(
		"git apply --allow-empty -v /tmp/gold_patch.diff",
		workdir="/testbed",
		user="root",
	)
	if _val.exit_code != 0:
		_val = container.exec_run(
			"patch --batch --fuzz=5 -p1 -i /tmp/gold_patch.diff",
			workdir="/testbed",
			user="root",
		)

	_git_diff_output_after = (
		container.exec_run("git diff", workdir="/testbed")
		.output.decode("utf-8")
		.strip()
	)
	test_output, _, _ = exec_run_with_timeout(
		container, "/bin/bash /eval.sh", timeout=180
	)
	post_gold_tests = MAP_REPO_TO_PARSER[instance.repo](test_output)

	diff_tests = {
		test: post_gold_tests[test]
		for test in post_gold_tests
		if post_gold_tests[test] != pre_gold_tests.get(test)
	}
	print(f"Diff tests: {diff_tests}")

	cleanup_container(client, container, logger)


def build_pre_tests(dataset: List[Dict]) -> None:
	"""Build images to run tests for instances in the specified dataset.

	Args:
		dataset (List[Dict]): A list of dictionaries, each containing
			information about an instance.
	"""
	client = docker.from_env()

	processed_dataset = [
		instance
		| {
			"PASS_TO_PASS": [],
			"FAIL_TO_PASS": [],
		}
		for instance in dataset
	]

	with open("/tmp/processed_dataset.json", "w") as f:
		json.dump(processed_dataset, f)
	swebench_dataset = load_swebench_dataset("/tmp/processed_dataset.json")
	os.remove("/tmp/processed_dataset.json")

	_, failed = build_env_images(client, swebench_dataset, force_rebuild=True)
	if len(failed) > 0:
		print(f"Failed to build images for instances: {failed}")
		return

	for instance_base, swe_instance in zip(
		processed_dataset, swebench_dataset, strict=True
	):
		_process_instance(
			make_test_spec(swe_instance),
			instance_base["test_patch"],
			instance_base["patch"],
			client,
		)


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python patch_tests.py <dataset_path>")
		sys.exit(1)

	dataset_path = sys.argv[1]

	with open(dataset_path, "r") as f:
		dataset = json.load(f)
	build_pre_tests(dataset)
