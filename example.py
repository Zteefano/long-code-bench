from openai import OpenAI
 
host = "127.0.0.1"
port = 8000 
api_key = "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"

client = OpenAI(
    base_url=f"http://{host}:{port}/v1",
    api_key=api_key,
)

completion = client.chat.completions.create(
  model="/leonardo_scratch/large/userinternal/mviscia1/models/Llama-3.1_405B-Instruct",
  messages=[
      {"role": "system", "content": "You are a helpful assistant built by Cineca to answer User's question about HPC."},
      {"role": "user", "content": "Can you write an example of sbatch file?"}
  ],
)
 
print(completion.choices[0].message.content)
