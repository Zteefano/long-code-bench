import hydra
import asyncio
from datasets import load_from_disk
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.long_code_bench.inference.hf_eval import DatasetsEvaluator
from src.long_code_bench.models import APIModel, Model, CinecaAPI

load_dotenv()

# Your asynchronous main function
async def main_async(cfg: DictConfig) -> None:
    dataset = load_from_disk(cfg.dataset.dataset_path)
    model: Model = instantiate(cfg.model)

    evaluator = DatasetsEvaluator(
        model,
        dataset,
        "text",
        cfg.output,
        batch_size=cfg.batch_size,
        max_context_length=100_000,
        max_output_length=None,
    )

    # If your model is asynchronous, use the async evaluator methods.
    if isinstance(model, CinecaAPI):
        await evaluator.async_run()
    elif isinstance(model, APIModel) and cfg.batch_queue:
        evaluator.run_batch_queue()
    else:
        evaluator.run()

# The Hydra-decorated synchronous main that calls your async main
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(main_async(cfg))

if __name__ == "__main__":
    main()
