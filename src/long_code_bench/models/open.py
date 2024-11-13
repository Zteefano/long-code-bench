from src.long_code_bench.models.base import Model


class OpenSourceModel(Model):
    """Class for all open-source models.

    An open-source model is one with piblicly available weights and must
    be run locally.

    Args:
        hf_path (str): The dataset's path on the Hugging Face Hub.
    """

    def __init__(self, hf_path: str) -> None:
        self.name = hf_path
        # TODO: Code for loading and running open models

    def generate(self, prompt: str, max_length: int = -1) -> str:
        """Generate text given a prompt.

        Args:
            prompt (str): The prompt to generate text from.
            max_length (int): The maximum length of the generated text.
                If `-1`, the model can generate text of any length. By
                default, `-1`.

        Returns:
            The generated text.
        """
        # TODO: Generate text, might need to be async
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the model, used for identification."""
        return self.name
