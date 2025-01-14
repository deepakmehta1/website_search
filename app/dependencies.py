from app.services.search_service import SearchService
from app.config import config
from pymongo import MongoClient

def get_search_service() -> SearchService:
    client = MongoClient(config.MONGO_URI)
    db = client[config.DB_NAME]
    search_service = SearchService(db)
    return search_service
