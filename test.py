import json
import os
from src.long_code_bench.models import APIModel, OpenSourceModel

# Load API keys from keys.json
with open("keys.json", "r") as f:
    keys = json.load(f)

# Prompt for testing
prompt = "Explain the significance of the Renaissance period."

# Specify model type and version
model_type = "llama"  #"anthropic" "llama" "openai"

# For OpenAI and Anthropic models, specify the model version
model_version = "gpt-3.5-turbo" if model_type == "openai" else "claude-2"  # "gpt-4" "claude-2"

# Specify the Hugging Face path for the desired LLaMA model
hf_path = "meta-llama/Llama-2-7b-hf"  # "meta-llama/Llama-2-13b-hf"

if model_type in ["openai", "anthropic"]:
    # Get the API key
    api_key = keys.get(model_type)
    if not api_key:
        raise ValueError(f"No API key found for model type: {model_type}")

    model = APIModel(model_type=model_type, model_version=model_version, api_key=api_key)

elif model_type == "llama":    
    # Get the API key
    hf_token = keys.get("huggingface")
    model = OpenSourceModel(hf_path=hf_path, token=hf_token)

else:
    raise ValueError(f"Unsupported model type: {model_type}")

# Generate and print text based on the prompt
try:
    response = model.generate(prompt)
    print("Generated Text:")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
