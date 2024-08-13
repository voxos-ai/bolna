from pydantic import BaseModel
from typing import Union, Optional

# Embedding config
class embeddings(BaseModel):
    provider:str
    embedding_model_name:Optional[str] = ""

# DB Configs
class LanceDBConfig(BaseModel):
    loc: Optional[str] = ""

class MongoDBConfig(BaseModel):
    uri: Optional[str] = ""
    db: Optional[str] = ""
    collection_name: Optional[str] = ""

# Provider Configs
class ProviderConfig(BaseModel):
    embedding_name:embeddings
    chunk_size:int
    overlapping:int
    worker:int
    similarity_top_k:int
    rag: Union[LanceDBConfig,MongoDBConfig]

# Rag Config
class RAGConfig(BaseModel):
    provider:str
    provider_config:ProviderConfig
    
class Query(BaseModel):
    provider:str
    index:str
    query:str

# Utility checks for RAG
class RAGTaskStatus:
    WAIT = "WAITING"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"
    SUCESSFUL = "SUCESSFUL"

class RAGTask(BaseModel):
    file_loc:str
    _status:str = RAGTaskStatus.WAIT
    _message:str = ""
    _index:str = ""
    _nodes:list = []
