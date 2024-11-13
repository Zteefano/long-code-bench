from src.long_code_bench.models.base import Model


class APIModel(Model):
    """Class for all models accessible through API calls.

    This class is used for closed models that are not open-source and
    cannot be run locally. They are accessed through an API, which
    requires a key and might have usage limits.
    """

    def __init__(self) -> None:
        # TODO: Code for initializing API models, focus on the API key
        pass

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
        # TODO: Generate text, handle usage limits
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the model, used for identification."""
        # TODO: Define best way to get the name of the model
        raise NotImplementedError
