import json
import os
import openai
from tqdm import tqdm
from dotenv import load_dotenv


def load_jsonl(file_path):
    with open(file_path, "r") as file:
        for line in file:
            yield json.loads(line)

def main():
    load_dotenv()  

    base_dir = "dataset/output_512K"
    result_dir = os.path.join(base_dir, "results")

    file_name = "Qwen2.5-14B-Instruct-1M.jsonl"

    input_file = os.path.join(result_dir, file_name)

    #Load Jsonl file
    dataset = list(load_jsonl(input_file))

    for entry in tqdm(dataset, desc="Evaluating prompts"):

        # entry.keys() = dict_keys(['prompt', 'generation', 'correct_letter', 'question', 'repo_text', 'prompt_goal', 'repo'])

        generation = entry["generation"]
        question = entry["question"]
        correct_answer = entry["correct_letter"]
        
        # Construct the prompt
        prompt = f"""
I will give you a question, that an AI model answered, and the correct answer. 
The question is: {question}
The AI model answered: {generation}
The correct answer is: {correct_answer}
Please tell me if the AI model answered correctly or not.
If the AI model answered correctly, please answer with "yes".
If the AI model answered incorrectly, please answer with "no".
"""

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(model = "gpt-4o",
                                                    messages = [{"role": "user", "content": prompt}],
                                                    temperature = 0.0)
        generation = response.choices[0].message.content
        entry["is_correct"] = generation

    # Save the modified dataset back to a JSONL file
    # Output file name = model_name + "-eval.jsonl"
    # Expect - in the model name
    # Example: AI21-Jamba-1.5-Large-eval.jsonl
    model_name = file_name[:-6]  # Remove the last 6 characters (".jsonl")
    output_file_name = f"{model_name}-eval.jsonl"
    output_file = os.path.join(result_dir, output_file_name)

    with open(output_file, "w") as file:
        for entry in dataset:
            file.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()