
from bolna.helpers.logger_config import configure_logger
from bolna.memory.cache.base_cache import BaseCache
import numpy as np
from fastembed import TextEmbedding
import numpy as np

logger = configure_logger(__name__)

class VectorCache(BaseCache):
    def __init__(self, index_provider = None, embedding_model = "BAAI/bge-small-en-v1.5"):
        super().__init__()
        self.index_provider = index_provider
        self.embedding_model = TextEmbedding(model_name=embedding_model)
    
    def set(self, documents):
        self.documents = documents
        self.embeddings  = list(
            self.embedding_model.passage_embed(documents)
        )
    
    def __get_top_cosine_similarity_doc(self, query_embedding):
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis]
        similarities = np.dot(embeddings_norm, query_norm)
        most_similar_index = np.argmax(similarities)
        return self.documents[most_similar_index]
        
    def get(self, query):
        if self.index_provider is None:
            query_embedding = list(self.embedding_model.query_embed(query))[0]
            response = self.__get_top_cosine_similarity_doc(query_embedding= query_embedding)
            return response
        else:
            logger.info("Other mechanisms not yet implemented")



    
    