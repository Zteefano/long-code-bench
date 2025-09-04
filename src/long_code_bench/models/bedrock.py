import json
import time
from typing import List, Optional, override

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from src.long_code_bench.models.api import APIModel

load_dotenv()


class BedrockAIModel(APIModel):
	"""A class to interact with the Bedrock API models.

	Args:
		model_version (str): The ID of the model.
		region (str): The AWS region where the model is hosted. By
			default, `"us-east-1"`.
	"""

	def __init__(
		self,
		model_version: str,
		region: str = "us-east-1",
	) -> None:
		self.model_version = model_version

		self.client = boto3.client(
			"bedrock-runtime",
			region_name=region,
			config=Config(retries={"max_attempts": 20, "mode": "standard"}),
		)

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
		if "llama" in self.model_version or "deepseek" in self.model_version:
			while True:
				try:
					native_request = {
						"prompt": prompt,
					}
					request = json.dumps(native_request)
					response = self.client.invoke_model(
						modelId=self.model_version, body=request
					)
					model_response = json.loads(response["body"].read())
					response_text = model_response["generation"]
					return response_text
				# except self.client.exceptions.ValidationException:
				# return ""
				except self.client.exceptions.ThrottlingException:
					time.sleep(60)
				except Exception as e:
					raise e

		while True:
			try:
				native_request = {
					"messages": [{"role": "user", "content": prompt}],
				}
				request = json.dumps(native_request)
				response = self.client.invoke_model(
					modelId=self.model_version, body=request
				)
				model_response = json.loads(response["body"].read())
				response_text = model_response["choices"][0]["message"][
					"content"
				]
				return response_text
			except self.client.exceptions.ValidationException:
				return ""
			except Exception as e:
				raise e

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

		This method is currently not implemented for the Bedrock API
		models.

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

		Raises:
			NotImplementedError: This method is not implemented for
				the Bedrock API models.

		Returns:
			List[str]: The list of generated texts.
		"""
		raise NotImplementedError(
			"This method is not implemented for the Bedrock API models."
		)

	@property
	@override
	def name(self) -> str:
		"""Name of the model, used for identification."""
		return self.model_version
