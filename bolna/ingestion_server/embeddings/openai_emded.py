from bolna.ingestion_server.embeddings.base import BaseEmbed
from llama_index.embeddings.openai import OpenAIEmbedding
import dotenv
import os

dotenv.load_dotenv()

class OpenAI(BaseEmbed):
    def __init__(self, model: str) -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            model (str): The model name to be used for embeddings.
        
        Raises:
            ValueError: If the OpenAI API key is not found in the environment variables.
        """
        super().__init__("OpenAI")
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI KEY IS NOT FOUND")
        self.embedding_instance = OpenAIEmbedding(model=model, api_key=api_key)

    def get_embedding(self):
        """Retrieve the embedding instance.

        Returns:
            OpenAIEmbedding: The instance of the OpenAI embedding.
        """
        return self.embedding_instance