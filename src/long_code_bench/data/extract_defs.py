import ast
import os
from typing import List, Literal, Tuple, Union

DefinitionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]


class DefVisitor(ast.NodeVisitor):
	"""Class to visit nodes keeping track of outer class names."""

	# Vedi se riesci a definire questa classe come un iteratore che
	# resituisce i nodi insieme al nome della classe esterna

	def __init__(self) -> None:
		self.current_class = None

	def visit_class_def(self, node):
		self.definitions.append((node.name, "class"))
		previous_class = self.current_class
		self.current_class = node.name
		self.generic_visit(node)
		self.current_class = previous_class

	def visit_FunctionDef(self, node):
		name = (
			f"{self.current_class}.{node.name}"
			if self.current_class
			else node.name
		)
		self.definitions.append((name, "func"))
		self.generic_visit(node)

	def visit_AsyncFunctionDef(self, node):
		name = (
			f"{self.current_class}.{node.name}"
			if self.current_class
			else node.name
		)
		self.definitions.append((name, "func"))
		self.generic_visit(node)


def extract_defs_from_file(
	file_path: str,
) -> List[Tuple[str, Literal["func", "class"]]]:
	"""Extract function and class definitions from a Python file.

	Args:
		file_path (str): Path to the Python file.

	Returns:
		List[Tuple[str, Literal["func", "class"]]]: List of function and
			class definitions, together with their type.
	"""
	definitions: List[Tuple[str, Literal["func", "class"]]] = []

	try:
		with open(file_path, "r", encoding="utf-8") as f:
			file_contents = f.read()

		tree = ast.parse(file_contents, filename=file_path)
		DefVisitor(definitions).visit(tree)
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
				[f"{full_path}::{def_name}" for def_name, _ in defs]
			)

	return output_lines
