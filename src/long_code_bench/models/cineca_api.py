import ast
from typing import Any, List, Optional, Dict

import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from openai import OpenAI
from src.long_code_bench.models.base import Model


def _convert_value(value: str) -> Any:  # noqa: ANN401
	try:
		return ast.literal_eval(value)
	except (ValueError, SyntaxError):
		return value


class CinecaAPI(Model):
	"""Class for all open-source models from the Hugging Face Hub.

	Args:
		hf_path (str): The model's path on the Hugging Face Hub.
		token (Optional[str]): The token to use for the model. By
			default, `None`.
		offline (bool): Whether to load the model and the tokenizer from
			local files only. By default, `False`.
		**kwargs: Additional keyword arguments to pass to the model.
			These argument can provide additional configuration like
			`attn_implementation`, or `torch_dtype`.
	"""
	def __init__(self, base_url: str, api_key: str, model: str):
		self.client = OpenAI(base_url=base_url, api_key=api_key)
		self.model = model
		self.tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-1.5-Large")
		self.max_window = 256000

	def generate(
		self,
		prompt: str,
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
	) -> str:
		"""Generate text given a prompt by making an API call.

		Args:
			prompt (str): The prompt to generate text from.
			max_context_length (Optional[int]): The maximum length of
				the context to consider. Only present for compatibility,
				but not used. By default, `None`.
			max_output_length (Optional[int]): The maximum length of the
				output text. Only present for compatibility, but not
				used. By default, `None`.

		Returns:
			str: The generated text.
		"""
		print(prompt)
		quit()
		response = self.client.chat.completions.create(
			messages=[{"role": "user", "content": prompt}],
			model=self.model,
			max_tokens=self.max_window,
		)
		return response.choices[0].message.content or ""

	def generate_batch(
		self,
		prompts: List[str],
		max_context_length: Optional[int] = None,
		max_output_length: Optional[int] = None,
		ids: Optional[List[str]] = None,
		file_name: Optional[str] = None,
		batch_size: int = 10,
	) -> List[str]:
		"""Generate text for a batch of prompts.

		The main feature of this method is that it generates text
		through batch API requests, which are handled asynchronously.
		This incurs a delay in the response time, but is more efficient
		in terms of costs.

		Args:
			prompts (List[str]): The list of prompts to generate text
				from.
			max_context_length (Optional[int]): The maximum length of
				the context to consider. Only present for compatibility,
				but not used. By default, `None`.
			max_output_length (Optional[int]): The maximum length of the
				output text. Only present for compatibility, but not
				used. By default, `None`.
			ids (Optional[List[str]]): The list of IDs associated to the
				prompts. Necessary to keep track of the generated text
				associated with each prompt. By default, `None`.
			file_name (Optional[str]): The file to store the generated
				results from a queued batch generation. If `None`, a
				temporary file is used that is then deleted. By default,
				`None`.
			batch_size (int): The size of the batch, _i.e._, the number
				of prompts to generate text for in each batch file. By
				default, `10`.

		Returns:
			List[str]: The list of generated texts.
		"""
		assert ids is not None, "IDs must be provided for batch generation"

		async def _generate_batches() -> List[str]:
			tasks = []
			for i in range(0, len(prompts), batch_size):
				prompts_batch = prompts[i : i + batch_size]
				ids_batch = ids[i : i + batch_size]
				tasks.append(self._create_batch_file(prompts_batch, ids_batch))

			results = await asyncio.gather(*tasks, return_exceptions=True)
			combined_results = {}
			for result in results:
				if not isinstance(result, BaseException):
					combined_results.update(result)

			to_return = []
			for instance_id in ids:
				to_return.append(combined_results.get(instance_id, ""))
			return to_return

		results = asyncio.run(_generate_batches())

		if file_name:
			with open(file_name, "w") as f:
				for result in results:
					f.write(result + "\n")

		return results

	async def _create_batch_file(
		self, prompts_batch: List[str], ids_batch: List[str]
	) -> Dict[str, str]:
		tmp_dir = os.getenv("TMPDIR", "/tmp")
		temp_file_name = os.path.join(
			tmp_dir, f"batch_file_{int(time.time())}_{os.getpid()}.jsonl"
		)

		with open(temp_file_name, "w") as f:
			for prompt, instance_id in zip(
				prompts_batch, ids_batch, strict=True
			):
				task = {
					"custom_id": f"task-{instance_id}",
					"method": "POST",
					"url": "/v1/chat/completions",
					"body": {
						"model": self.model_version,
						"temperature": 0.1,
						"messages": [
							{"role": "user", "content": prompt},
						],
					},
				}
				f.write(json.dumps(task) + "\n")

		batch_file = self.client.files.create(
			file=open(temp_file_name, "rb"), purpose="batch"
		)
		batch_job = self.client.batches.create(
			input_file_id=batch_file.id,
			endpoint="/v1/chat/completions",
			completion_window="24h",
		)

		while True:
			job_status = self.client.batches.retrieve(batch_job.id)
			if (
				job_status.status == "completed" and job_status.output_file_id
			) or job_status.status == "failed":
				break
			await asyncio.sleep(180)

		result_file_id = batch_job.output_file_id
		if not result_file_id:
			raise ValueError("Batch job failed to generate output")
		result = self.client.files.content(result_file_id).content

		result_file_name = os.path.join(
			tmp_dir, f"batch_job_results_{os.getpid()}.jsonl"
		)
		with open(result_file_name, "wb") as file:
			file.write(result)

		results = {}
		with open(result_file_name, "r") as file:
			for line in file:
				json_object = json.loads(line.strip())
				results[json_object["task_id"][5:]] = json_object["response"][
					"body"
				]["choices"][0]["message"]["content"]

		os.remove(temp_file_name)

		return results

	@property
	def name(self) -> str:
		"""Name of the model, used for identification."""
		return self.model