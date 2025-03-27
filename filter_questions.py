import json
import os
import openai
from tqdm import tqdm
from dotenv import load_dotenv

def evaluate_prompts(input_path, output_path):
    with open(input_path, "r") as file:
        dataset = json.load(file)

    for entry in tqdm(dataset, desc="Evaluating prompts"):
        prompt = entry["prompt"]
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(model = "gpt-4o",
                                                  messages = [{"role": "user", "content": prompt}],
                                                  temperature = 0.0)
        generation = response.choices[0].message.content
        entry["is_hard"] = generation

    with open(output_path, "w") as file:
        json.dump(dataset, file, indent=4)

def main():
    load_dotenv()  # Load environment variables from .env
    
    base_dir = "output_64K"

    input_file = os.path.join(base_dir, "dataset_filtering.json")

    output_file = os.path.join(base_dir, "filtered_dataset.json")

    # Set your OpenAI API key (this is the only keyword argument allowed)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set your OPENAI_API_KEY environment variable.")

    evaluate_prompts(input_file, output_file)

if __name__ == "__main__":
    main()
