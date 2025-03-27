import json
import os
import boto3
from tqdm import tqdm
from transformers import AutoTokenizer

def load_dataset(file_path: str):
    """Load the dataset from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def count_processed_lines(results_file: str) -> int:
    """Count the number of non-empty lines already written in the results file."""
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return sum(1 for line in f if line.strip())
    return 0

def process_prompts(dataset, results_file: str, model_id: str):
    """
    Process each prompt in the dataset:
      - Skip any prompts that have been already processed.
      - For each new prompt, call the Amazon Bedrock API.
      - Save the response (with prompt and additional metadata) as a new JSON line.
    """
    processed_count = count_processed_lines(results_file)
    total_instances = len(dataset)
    remaining = total_instances - processed_count
    print(f"Total prompts: {total_instances}. Already processed: {processed_count}. Remaining: {remaining}")

    # Initialize the Bedrock runtime client.
    client = boto3.client("bedrock-runtime")

    progress_bar = tqdm(total=remaining, desc="Processing prompts")

    # Open the results file in append mode.
    with open(results_file, "a") as f_results:
        for i, entry in enumerate(dataset):
            # Skip prompts that have already been processed.
            if i < processed_count:
                continue

            repo_text = entry.get("repo_text", "")
            question = entry.get("question", "")
            prompt_text = (
                "You are a coding expert. Your task is to analyze a GitHub repository and then answer one question about it.\n"
                f"Repository:\n{repo_text}\nQuestion:\n{question}\n"
                "Please analyze the repository text, reason through the question, and then choose among A, B, C, D answers the correct one.\n"
                "Provide your final answer in the following format:\nFinal Answer: <LETTER>"
            )
            entry["prompt"] = prompt_text

            # Prepare the payload for the model.
            payload = {
                "prompt": prompt_text,
                "max_tokens_to_sample": 100,  # Adjust as necessary.
                "temperature": 1.0,           # Adjust as needed.
            }
            payload_json = json.dumps(payload)

            try:
                response = client.invoke_model(
                    ModelId=model_id,
                    ContentType="application/json",
                    Body=payload_json,
                )
                # Read and decode the response.
                response_body = response["Body"].read().decode("utf-8")
                result_data = json.loads(response_body)
                # Adjust the key based on your model's output; here we assume "generated_text".
                generation = result_data.get("generated_text", "")

                # Prepare the result record, including additional metadata if available.
                output = {
                    "prompt": prompt_text,
                    "generation": generation,
                    "question": entry.get("question"),
                    "correct_letter": entry.get("correct_letter"),
                    "repo_text": entry.get("repo_text"),
                    "prompt_goal": entry.get("prompt_goal"),
                    "repo": entry.get("repo"),
                }
                f_results.write(json.dumps(output) + "\n")
                f_results.flush()
            except Exception as e:
                print(f"Error processing prompt index {i}: {e}")
            progress_bar.update(1)
    progress_bar.close()

def main():
    # Define the dataset and results file paths.
    file_path = "dataset/output_1M/final_dataset.json"
    results_dir = "dataset/output_1M/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "amazon_bedrock_results.jsonl")

    # Load the dataset.
    dataset = load_dataset(file_path)
    print(f"Loaded {len(dataset)} prompts")

    # Initialize the tokenizer (using the same model as before).
    tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Large-1.5")
    max_length = 1009000  # Define your maximum token length.

    # Filter prompts based on token length.
    kept_prompts = []
    for i, entry in enumerate(dataset):
        # Build the prompt if not already present.
        prompt_text = entry.get("prompt")
        if prompt_text is None:
            repo_text = entry.get("repo_text", "")
            question = entry.get("question", "")
            prompt_text = (
                "You are a coding expert. Your task is to analyze a GitHub repository and then answer one question about it.\n"
                f"Repository:\n{repo_text}\nQuestion:\n{question}\n"
                "Please analyze the repository text, reason through the question, and then choose among A, B, C, D answers the correct one.\n"
                "Provide your final answer in the following format:\nFinal Answer: <LETTER>"
            )
            entry["prompt"] = prompt_text

        tokenized = tokenizer(prompt_text)["input_ids"]
        if len(tokenized) < max_length:
            kept_prompts.append(entry)
            print(f"Kept instance {i} with token length {len(tokenized)}")
        else:
            print(f"Discarded instance {i} with token length {len(tokenized)}")
    dataset = kept_prompts
    print("Kept", len(dataset), "prompts after filtering by token length")

    # Specify the model ID for Amazon Bedrock.
    model_id = "amazon.titan-tgi"  # Replace with your actual model ID if needed.

    # Process prompts and write results.
    process_prompts(dataset, results_file, model_id)
    print(f"Results written to {results_file}")

if __name__ == "__main__":
    main()
