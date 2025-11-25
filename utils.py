from pymongo import MongoClient
import pandas as pd


MONGO_URI = "mongodb+srv://new:new@cluster0.psu3sng.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "tr_generator_db"
COLLECTION_NAME = "transformations"


client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def insert_dataframe(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    result = collection.insert_many(records)
    return len(result.inserted_ids)

def get_sample_records(limit=10):
    return list(collection.find({}, {"_id": 0}).limit(limit))