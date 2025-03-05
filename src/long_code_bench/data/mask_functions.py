import ast
import os
import random
import shutil
from collections import defaultdict
from typing import List, Literal, Tuple, Union

DefinitionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]


def find_definition_nodes(
	source: str, target_names: set
) -> List[Tuple[DefinitionNode, Literal["function", "class"]]]:
	"""Find function and class definition nodes in the source code.

	Args:
		source (str): Source code text.
		target_names (set): Set of target function and class names to
			find.

	Returns:
		List[Tuple[DefinitionNode, Literal["function", "class"]]]: List
			of tuples containing the definition node and its type (
			either `"function"` or `"class"`).
	"""
	nodes = []

	try:
		tree = ast.parse(source)
		for node in ast.walk(tree):
			if (
				isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
				and node.name in target_names
			):
				nodes.append((node, "function"))
			elif isinstance(node, ast.ClassDef) and node.name in target_names:
				nodes.append((node, "class"))
	except Exception as e:
		print(f"Error parsing source: {e}")

	return nodes


def mask_definition_in_source(
	source_lines: List[str],
	node: DefinitionNode,
	def_type: Literal["function", "class"],
) -> List[str]:
	"""Mask a function or class definition in the source code.

	Args:
		source_lines (List[str]): List of source code lines.
		node (DefinitionNode): Function or class definition node.
		def_type (Literal["function", "class"]): Type of definition to
			mask.

	Raises:
		ValueError: If an invalid definition type is provided.

	Returns:
		List[str]: List of source code lines with the definition masked.
	"""
	start_idx = node.lineno - 1
	end_idx = node.end_lineno - 1

	header_end_idx = start_idx
	for i in range(start_idx, end_idx + 1):
		if source_lines[i].strip().endswith(":"):
			header_end_idx = i
			break

	header_line = source_lines[start_idx]
	header_indent = len(header_line) - len(header_line.lstrip())
	body_indent = " " * (header_indent + 4)

	if def_type == "function":
		stub_body = [
			f"{body_indent}# FILL HERE\n",
			f"{body_indent}return None\n",
		]
	elif def_type == "class":
		stub_body = [f"{body_indent}# FILL HERE\n", f"{body_indent}pass\n"]
	else:
		raise ValueError(f"Invalid definition type: {def_type}")

	new_block = source_lines[start_idx : header_end_idx + 1] + stub_body
	new_source_lines = (
		source_lines[:start_idx] + new_block + source_lines[end_idx + 1 :]
	)
	return new_source_lines


def mask_random_definitions(
	definitions: List[str], num_to_mask: int = 5
) -> None:
	"""Mask random definitions in the provided list of files.

	Args:
		definitions (List[str]): List of definitions in the format
			"file_path::def_name".
		num_to_mask (int, optional): Number of definitions to mask.
			Defaults to 5.
	"""
	selected_entries = random.sample(
		definitions, min(num_to_mask, len(definitions))
	)

	file_to_defs = defaultdict(list)
	for entry in selected_entries:
		try:
			file_path, def_name = entry.split("::")
			file_to_defs[file_path].append(def_name)
		except ValueError:
			print(f"Skipping invalid entry format: {entry}")

	for file_path, def_names in file_to_defs.items():
		if not os.path.exists(file_path):
			print(f"File {file_path} does not exist. Skipping.")
			continue

		backup_path = file_path + ".bak"
		if not os.path.exists(backup_path):
			shutil.copy2(file_path, backup_path)
			print(f"Backup created for {file_path} as {backup_path}")
		else:
			print(f"Backup already exists for {file_path}")

		with open(file_path, "r", encoding="utf-8") as f:
			source_lines = f.readlines()
		source_text = "".join(source_lines)

		nodes = find_definition_nodes(source_text, set(def_names))
		if not nodes:
			print(
				f"No matching definitions found in {file_path} for {def_names}"
			)
			continue

		nodes.sort(key=lambda x: x[0].lineno, reverse=True)
		modified_lines = source_lines
		for node, def_type in nodes:
			modified_lines = mask_definition_in_source(
				modified_lines, node, def_type
			)

		with open(file_path, "w", encoding="utf-8") as f:
			f.write("".join(modified_lines))
		masked_names = ", ".join(node.name for node, _ in nodes)
		print(f"Masked definitions in {file_path}: {masked_names}")


if __name__ == "__main__":
	# Example usage
	definitions_list = [
		"file1.py::function1",
		"file2.py::class1",
		# Add more definitions as needed
	]
	mask_random_definitions(definitions_list, num_to_mask=1)
