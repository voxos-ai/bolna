import os
import pymongo
import pprint
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from bolna.rag.base_class import DatabaseConnector, VectorSearchEngine, run_queries

class MongoDBConnector(DatabaseConnector):
    def connect(self):
        try:
            self.client = pymongo.MongoClient(self.connection_string)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            print("Connection to MongoDB successful")
        except pymongo.errors.ConnectionFailure as e:
            print(f"Connection failed: {e}")

    def disconnect(self):
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB")

    def verify_data(self):
        doc_count = self.collection.count_documents({})
        print(f"Total documents in collection: {doc_count}")
        if doc_count > 0:
            print("Sample document:")
            pprint.pprint(self.collection.find_one())
        else:
            print("No documents found in the collection.")

    def setup_vector_store(self):
        return MongoDBAtlasVectorSearch(
            self.client, 
            db_name=self.db_name, 
            collection_name=self.collection_name, 
            index_name="vector_index"
        )

def main():
    # Configuration
    OPENAI_API_KEY = "***"
    MONGO_URI = "***"
    DB_NAME = "***"
    COLLECTION_NAME = "***"

    # Set OpenAI API Key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Create MongoDB connector
    mongo_connector = MongoDBConnector(MONGO_URI, DB_NAME, COLLECTION_NAME)
    mongo_connector.connect()
    mongo_connector.verify_data()

    # Set up search engine
    search_engine = VectorSearchEngine(mongo_connector)
    search_engine.setup_llama_index()
    search_engine.create_index()

    # Example queries
    queries = [
        "Recommend a romantic movie suitable for the Christmas season",
        "What are some popular action movies from the 1990s?",
        "Can you suggest a family-friendly animated film?",
        "Tell me about a science fiction movie with time travel",
        "What's a good comedy movie from the 2000s?"
    ]

    # Run queries
    run_queries(search_engine, queries)

    # Disconnect
    mongo_connector.disconnect()

if __name__ == "__main__":
    main()