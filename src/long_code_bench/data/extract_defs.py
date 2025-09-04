import ast
import os
from typing import List


def extract_defs_from_file(file_path: str) -> List[str]:
	"""Extract function and class definitions from a Python file.

	Args:
		file_path (str): Path to the Python file.

	Returns:
		List[str]: List of function and class definitions.
	"""
	definitions: List[str] = []

	try:
		with open(file_path, "r", encoding="utf-8") as f:
			file_contents = f.read()

		tree = ast.parse(file_contents, filename=file_path)
		for node in ast.walk(tree):
			if isinstance(
				node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
			):
				definitions.append(node.name)
	except Exception as e:
		print(f"Error processing {file_path}: {e}")

	return definitions


def extract_defs_from_dir(base_dir: str) -> List[str]:
	"""Extract function and class definitions from a directory.

	This function extracts definitions from all the Python files inside
	a directory recursively.

	Args:
		base_dir (str): Path to the directory.

	Returns:
		List[str]: List of function and class definitions.
	"""
	output_lines: List[str] = []

	for root, _, files in os.walk(base_dir):
		for file in files:
			if not file.endswith(".py"):
				continue

			full_path = os.path.join(root, file)
			defs = extract_defs_from_file(full_path)
			output_lines.extend(
				[f"{full_path}::{def_name}" for def_name in defs]
			)

	return output_lines
