from abc import ABC, abstractmethod


class Model(ABC):
    """Base class for all language models.

    A language model is a model that can generate text, given a prompt.
    Moreover, the model needs to have a name, which is used for
    identification.
    """

    @abstractmethod
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
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model, used for identification."""
        raise NotImplementedError
