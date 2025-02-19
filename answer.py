import os
import json
import logging
from openai import OpenAI
from tqdm import tqdm


host = "127.0.0.1"
port = 8000
api_key = "password"
model = "/leonardo/home/userexternal/lromani0/llm_models/AI21-Jamba-1.5-Large"
input_file = "combined_prompt.json"
output_file = "/Users/lucaromani/Lavoro/Datasets/results.json"

# Initialize the OpenAI-compatible client.
client = OpenAI(
    base_url=f"http://{host}:{port}/v1",
    api_key=api_key,
)

with open(input_file, "r") as f:
    prompts = json.load(f)

prompt = prompts[0]

completion = client.chat.completions.create(
  model=model,
  messages=[
      {"role": "system", "content": "You are a helpful assistant built by Cineca to answer User's question about HPC."},
        {"role": "user", "content": prompt},
  ],
)

# Save content to output file
with open(output_file, "w") as f:
    json.dump(completion.choices[0].message, f)

