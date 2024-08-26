from abc import ABC, abstractmethod

class BaseEmbed(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def get_embedding(self):
        """Method to retrieve the embedding representation."""
        raise NotImplementedError("Subclasses must implement this method.")