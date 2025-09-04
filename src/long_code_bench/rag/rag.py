import re
from abc import ABC, abstractmethod
from typing import Dict, List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGPreProcessor(ABC):
	"""Abstract base class for RAG preprocessing of prompts."""

	@abstractmethod
	def preprocess(self, prompt: str) -> Dict[str, str | List[Document]]:
		"""Preprocess the input prompt for RAG.

		Args:
			prompt (str): The input prompt to preprocess.

		Returns:
			Dict[str, str | List[Document]]: A dictionary containing the
				documents in the `"documents"` key, the query in the
				`"query"` key, and the text to append at the end of the
				prompt in the `"post_prompt"` key.
		"""
		raise NotImplementedError

	@abstractmethod
	def build_prompt(self, contents: Dict[str, str | List[Document]]) -> str:
		"""Build a prompt from RAG extracted contents.

		Args:
			contents (Dict[str, str | List[Document]]): The RAG
				extracted contents, which include a query under the
				`"query"` key, documents under the `"documents"` key,
				and the text to append at the end of the prompt under
				the `"post_prompt"` key.

		Returns:
			str: The constructed prompt.
		"""
		raise NotImplementedError


class RAGSWEBenchPreProcessor(RAGPreProcessor):
	"""RAG preprocessing for SWEBench prompts."""

	def preprocess(self, prompt: str) -> Dict[str, str | List[Document]]:
		"""Preprocess the input prompt for RAG.

		Args:
			prompt (str): The input prompt to preprocess.

		Returns:
			Dict[str, str | List[Document]]: A dictionary containing the
				documents in the `"documents"` key, the query in the
				`"query"` key, and the text to append at the end of the
				prompt in the `"post_prompt"` key.
		"""
		code_pattern = r"<code>(.*?)</code>"
		code_match = re.search(code_pattern, prompt, re.DOTALL)

		if code_match:
			code_content = code_match.group(1).strip()
			text_before = prompt[: code_match.start()].strip()
			text_after = prompt[code_match.end() :].strip()

			file_pattern = r"(\[start of (.+?)\].*?\[end of .+?\])"
			file_matches = re.findall(file_pattern, code_content, re.DOTALL)
			documents = [
				Document(
					id=file_name.strip(), page_content=file_content.strip()
				)
				for file_content, file_name in file_matches
			]
		else:
			code_content = ""
			text_before = prompt.strip()
			text_after = ""
			documents = [Document(id="full_prompt", page_content=prompt)]

		return {
			"documents": documents,
			"query": text_before,
			"post_prompt": text_after,
		}

	def build_prompt(self, contents: Dict[str, str | List[Document]]) -> str:
		"""Build a prompt from RAG extracted contents.

		Args:
			contents (Dict[str, str | List[Document]]): The RAG
				extracted contents, which include a query under the
				`"query"` key, documents under the `"documents"` key,
				and the text to append at the end of the prompt under
				the `"post_prompt"` key.

		Returns:
			str: The constructed prompt.
		"""
		documents = contents["documents"]
		query = contents["query"]
		post_prompt = contents["post_prompt"]

		if not documents:
			return f"{query} {post_prompt}".strip()

		documents_str = "\n\n".join(doc.page_content for doc in documents)
		return (
			f"{query}\n\n<code>{documents_str}</code>\n\n{post_prompt}".strip()
		)


class RAGQAPreProcessor(RAGPreProcessor):
	"""RAG preprocessing for QA prompts."""

	def preprocess(self, prompt: str) -> Dict[str, str | List[Document]]:
		"""Preprocess the input prompt for RAG.

		Args:
			prompt (str): The input prompt to preprocess.

		Returns:
			Dict[str, str | List[Document]]: A dictionary containing the
				documents in the `"documents"` key, the query in the
				`"query"` key, and the text to append at the end of the
				prompt in the `"post_prompt"` key.
		"""
		code_pattern = r"Repository:(.*?)Question:"
		code_match = re.search(code_pattern, prompt, re.DOTALL)

		if code_match:
			code_content = code_match.group(1).strip()
			text_before = prompt[: code_match.start()].strip()
			text_after = prompt[code_match.end() :].strip()

			file_pattern = r"(\[start of (.+?)\].*?\[end of .+?\])"
			file_matches = re.findall(file_pattern, code_content, re.DOTALL)
			documents = [
				Document(
					id=file_name.strip(), page_content=file_content.strip()
				)
				for file_content, file_name in file_matches
			]
		else:
			code_content = ""
			text_before = prompt.strip()
			text_after = ""
			documents = [Document(id="full_prompt", page_content=prompt)]

		new_documents = []
		toekenizer = tiktoken.encoding_for_model("gpt-4o")
		for doc in documents:
			if len(toekenizer.encode(doc.page_content)) < 50_000:
				new_documents.append(doc)
				continue
			splitter = RecursiveCharacterTextSplitter(
				chunk_size=50_000,
				chunk_overlap=10_000,
			)
			split_docs = splitter.split_text(doc.page_content)
			for i, split_doc in enumerate(split_docs):
				new_documents.append(
					Document(
						id=f"{doc.id}_part_{i}",
						page_content=split_doc,
					)
				)

		return {
			"documents": new_documents,
			"query": text_after,
			"post_prompt": text_before,
		}

	def build_prompt(self, contents: Dict[str, str | List[Document]]) -> str:
		"""Build a prompt from RAG extracted contents.

		Args:
			contents (Dict[str, str | List[Document]]): The RAG
				extracted contents, which include a query under the
				`"query"` key, documents under the `"documents"` key,
				and the text to append at the end of the prompt under
				the `"post_prompt"` key.

		Returns:
			str: The constructed prompt.
		"""
		documents = contents["documents"]
		query = contents["query"]
		post_prompt = contents["post_prompt"]

		if not documents:
			return f"{query} {post_prompt}".strip()

		documents_str = "\n\n".join(doc.page_content for doc in documents)
		return f"{post_prompt}\nRepository:\n{documents_str}\n{query}".strip()  # noqa: E501
