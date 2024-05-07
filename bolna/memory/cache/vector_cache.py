
from bolna.helpers.logger_config import configure_logger
from bolna.memory.cache.base_cache import BaseCache
from typing import List
import numpy as np
from fastembed import TextEmbedding

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
        scores = np.dot(self.embeddings, query_embedding)
        sorted_scores = np.argsort(scores)[::-1]
        return self.documents[sorted_scores[0]]
        
    def get(self, query):
        if self.index_provider is None:
            query_embedding = list(self.embedding_model.query_embed(query))[0]
            response = self.__get_top_cosine_similarity_doc(query_embedding= query_embedding)
            return response
        else:
            logger.info("Other mechanisms not yet implemented")



    
    