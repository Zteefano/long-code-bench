import asyncio
from src.long_code_bench.models.api import APIModel
from src.long_code_bench.inference.hf_eval import DatasetsEvaluator
from transformers import AutoTokenizer

async def main():
    # Configuration parameters:
    base_url = "http://127.0.0.1:8000/v1"   # vLLM API endpoint
    api_key = "password"  # Must match what vLLM expects
    # model_path = "/leonardo/home/userexternal/lromani0/llm_models/AI21-Jamba-1.5-Large"
    model_path= '/leonardo_scratch/large/userinternal/mviscia1/models/Llama-3.1_405B-Instruct'
    dataset_path = "/Users/lucaromani/Lavoro/Datasets/swebench_ver_tuned_small" 
    prompt_feature = "text"                    # The key in your dataset containing the prompt
    results_file = "results.jsonl"
    batch_size = 4                           # Adjust as needed
    system_prompt = "You are a helpful assistant built by Cineca to answer User's question about HPC."

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-1.5-Large")

    # Create the APIModel instance.
    model = APIModel(base_url=base_url, api_key=api_key, model=model_path)
    
    # Create the evaluator instance.
    evaluator = DatasetsEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        prompt_feature=prompt_feature,
        results_file=results_file,
        batch_size=batch_size,
        system_prompt=system_prompt,
        max_context_length=100000,
    )

    # Await the asynchronous run method.
    await evaluator.run()

if __name__ == "__main__":
    asyncio.run(main())
