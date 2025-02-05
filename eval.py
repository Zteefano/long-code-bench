from src.long_code_bench.models.api import APIModel
from src.long_code_bench.inference.hf_eval import DatasetsEvaluator

def main():
    # Configuration parameters:
    base_url = "http://127.0.0.1:8000/v1"   # vLLM API endpoint
    api_key = "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"  # Must match what vLLM expects
    model_path = "/leonardo_scratch/large/userinternal/mviscia1/models/Llama-3.1_405B-Instruct"
    dataset_path = "/leonardo/home/userexternal/lromani0/IscrC_TfG/datasets/swebench_ver_tuned_small" 
    prompt_feature = "text"                    # The key in your dataset containing the prompt
    results_file = "results.jsonl"
    batch_size = 4                           # Adjust as needed
    system_prompt = "You are a helpful assistant built by Cineca to answer User's question about HPC."

    # Initialize the API model.
    model = APIModel(base_url=base_url, api_key=api_key, model=model_path)

    # Create and run the evaluator.
    evaluator = DatasetsEvaluator(
        model=model,
        dataset_path=dataset_path,
        prompt_feature=prompt_feature,
        results_file=results_file,
        batch_size=batch_size,
        system_prompt=system_prompt,
    )
    evaluator.run()

if __name__ == "__main__":
    main()
