import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME = os.getenv("MONGO_DB", "website_search")
    HTML_DATA_COLLECTION = os.getenv("MONGO_HTML_DATA_COLLECTION", "html_data")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = Config()
