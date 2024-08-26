
## LanceDB 

{
    "provider": "LanceDB",
    "provider_config": {
        "embedding_name": {
            "provider": "OpenAI",
            "embedding_model_name": "text-embedding-3-small"
        },
        "chunk_size": 512,
        "overlapping": 200,
        "worker": 2,
        "similarity_top_k": 2,
        "rag": {
            "loc": "dev"
        }
    }
}