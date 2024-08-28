from rags.lancedb_rag import LanceDB
from rags.mongoDB_rag import MongoDB
from rags.base import BaseRAG

from typing import Dict

RAGProviders : Dict[str, BaseRAG] = {
    "LanceDB": LanceDB,
    "MongoDB": MongoDB
}