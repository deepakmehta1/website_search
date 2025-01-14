from fastapi import FastAPI, Depends
from pydantic import BaseModel
from app.models import ParseRequest, SearchRequest
from app.services.search_service import SearchService
from app.dependencies import get_search_service

app = FastAPI()

@app.post("/parse")
async def parse_url(request: ParseRequest, search_service: SearchService = Depends(get_search_service)):
    # Fetch, parse, and tokenize HTML content from the provided URL
    html_content = search_service.fetch_html(request.url)
    parsed_text = search_service.parse_html(html_content)
    chunks = search_service.tokenize_and_store(parsed_text, request.url)  # Pass URL to the service
    return {"chunks": chunks}

class SearchRequest(BaseModel):
    query: str
    url: str

@app.post("/search")
async def search(request: SearchRequest, search_service: SearchService = Depends(get_search_service)):
    """
    Perform semantic search using MongoDB and filter by URL.
    Accepts the query and URL in the request body.
    """
    results = search_service.search(request.query, request.url)
    return {"results": results}