from openai import OpenAI
import asyncio

class APIModel:
    def __init__(self, base_url: str, api_key: str, model: str):
        # Initialize the OpenAI client to talk to vLLM.
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
    
    async def process_prompt(self, prompt: str, system_prompt: str = None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use the async version of the OpenAI API.

        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages
        )

        return completion.choices[0].message


    async def process_batch(self, prompts, system_prompt=None):
        """
        For each prompt in the list, call the vLLM server via the OpenAI API.
        Optionally, include a system prompt.
        Returns a list of generated responses.
        """
        # results = []
        # for prompt in prompts:
        #     messages = []
        #     if system_prompt:
        #         messages.append({"role": "system", "content": system_prompt})
        #     messages.append({"role": "user", "content": prompt})
            
        #     completion = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=messages,
        #     )
        #     # Depending on your vLLM server, the response structure may vary.
        #     results.append(completion.choices[0].message)

        tasks = [self.process_prompt(prompt, system_prompt) for prompt in prompts]

        results = await asyncio.gather(*tasks)
        return results
