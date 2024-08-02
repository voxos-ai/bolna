
import pymongo

def get_mongo_client(mongo_uri):
  """Establish connection to the MongoDB."""
  try:
    client = pymongo.MongoClient(mongo_uri)
    print("Connection to MongoDB successful")
    return client
  except pymongo.errors.ConnectionFailure as e:
    print(f"Connection failed: {e}")
    return None

mongo_uri = "mongodb+srv://vipul:qqgr4bwAYl5pZSU9@testing-rag.nqaknom.mongodb.net/"

if not mongo_uri:
  print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME="movies"
COLLECTION_NAME="movies_records"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
print(type(db))
print(type(collection))
