import hydra
from datasets import load_from_disk
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.long_code_bench.inference.hf_eval import DatasetsEvaluator
from src.long_code_bench.models import APIModel, Model

load_dotenv()


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
	"""Run evaluation on a dataset using a model.

	Args:
		cfg (DictConfig): The configuration object.
	"""

	print("Loading dataset...")
	dataset = load_from_disk(cfg.dataset.dataset_path)

	print("Instantiating model...")
	model: Model = instantiate(cfg.model)

	evaluator = DatasetsEvaluator(
		model,
		dataset,
		"text",
		cfg.output,
		batch_size=cfg.batch_size,
		max_context_length=256000,
		max_output_length=None,
	)

	print("Running evaluation...")
	if isinstance(model, APIModel) and cfg.batch_queue:
		evaluator.run_batch_queue()
	else:
		evaluator.run()


if __name__ == "__main__":
	main()