import asyncio
from typing import List, Optional
from openai import OpenAI
from src.long_code_bench.models import Model
from transformers import AutoTokenizer


class CinecaAPI(Model):
    def __init__(self, base_url: str, api_key: str, model: str):
        # Initialize the OpenAI client to talk to vLLM.
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-1.5-Large")

    @property
    def name(self) -> str:
        return "CinecaAPI"

    async def process_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_context_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
    ) -> str:
        # Enforce max_context_length by tokenizing and truncating the prompt.
        if max_context_length is not None:
            tokenized = self.tokenizer(prompt)
            if len(tokenized["input_ids"]) > max_context_length:
                truncated_ids = tokenized["input_ids"][:max_context_length]
                prompt = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        api_kwargs = {}
        if max_output_length is not None:
            api_kwargs["max_tokens"] = max_output_length
        
        print(messages)
        quit()

        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            **api_kwargs
        )
        
        generated = completion.choices[0].message
        
        # Alternatively, if the API does not enforce a max output length,
        # you can post-process the generated text.
        if max_output_length is not None:
            # Tokenize the generated text and truncate if needed.
            gen_tokenized = self.tokenizer(generated)
            if len(gen_tokenized["input_ids"]) > max_output_length:
                truncated_ids = gen_tokenized["input_ids"][:max_output_length]
                generated = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        
        return generated

    async def process_batch(
        self, 
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_context_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
    ) -> List[str]:
        # Create tasks for each prompt, passing along the limits.
        tasks = [
            self.process_prompt(prompt, system_prompt, max_context_length, max_output_length)
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def generate(
        self,
        prompt: str,
        max_context_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        return await self.process_prompt(prompt, system_prompt, max_context_length, max_output_length)

    async def generate_batch(
        self,
        prompts: List[str],
        max_context_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
        system_prompt: Optional[str] = None,
        ids: Optional[List[str]] = None,
        file_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        return await self.process_batch(prompts, system_prompt, max_context_length, max_output_length)
