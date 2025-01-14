from pydantic import BaseModel

class ParseRequest(BaseModel):
    url: str

class SearchRequest(BaseModel):
    query: str
