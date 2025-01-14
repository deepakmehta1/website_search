import openai
from typing import List
from pymongo.collection import Collection
from app.config import config
from bs4 import BeautifulSoup
import httpx
import numpy as np
from openai import OpenAIError

# Set the OpenAI API key
openai.api_key = config.OPENAI_API_KEY

class SearchService:
    def __init__(self, db):
        """
        Public attribute: database collection
        """
        self.db: Collection = db[config.HTML_DATA_COLLECTION]  # Use 'html_data' collection

    def fetch_html(self, url: str) -> str:
        """
        Public method: Fetch HTML content from the URL.
        """
        response = httpx.get(url)
        if response.status_code != 200:
            raise ValueError("Failed to fetch the URL")
        return response.text

    def parse_html(self, html_content: str) -> str:
        """
        Public method: Parse HTML content using BeautifulSoup and clean it by removing unwanted tags.
        Only the DOM content is kept (i.e., everything inside <body>, <div>, <p>, <h1>, etc.)
        """
        soup = BeautifulSoup(html_content, "lxml")
        
        # Remove unwanted tags like <script>, <style>, and <head>
        for unwanted_tag in soup(["script", "style", "head"]):
            unwanted_tag.decompose()  # Remove these tags

        # Return the remaining content (keeping only the DOM)
        return str(soup)

    def tokenize_and_store(self, html_content: str, url: str) -> List[str]:
        """
        Public method: Tokenize the full HTML content and store the DOM structure in chunks in the database.
        """
        cleaned_html = self.parse_html(html_content)  # Clean the HTML by removing unwanted tags
        chunks = self._chunk_html(cleaned_html, max_tokens=500)
        self._index_chunks(chunks, url)
        return chunks

    def search(self, query: str, url: str, top_n: int = 10) -> List[str]:
        """
        Public method: Perform semantic search using MongoDB and filter by URL.
        """
        query_vector = self._create_vector(query)
        results_with_scores = []

        cursor = self.db.find({"url": url}, {"content": 1, "vector": 1})

        for document in cursor:
            doc_vector = document.get("vector")
            if doc_vector:
                score = self._cosine_similarity(np.array(query_vector), np.array(doc_vector))
                results_with_scores.append((document['content'], score))

        # Sort by cosine similarity in descending order
        results_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results based on similarity
        return [result[0] for result in results_with_scores[:top_n]]

    def _chunk_html(self, html_content: str, max_tokens: int) -> List[str]:
        """
        Private method: Split the entire HTML content (with tags and attributes) into chunks of max_tokens length,
        ensuring that chunks do not break HTML structure and close tags properly.
        """
        try:
            # Generate a vector for the entire HTML content to estimate its token count
            response = openai.Embedding.create(input=[html_content], model="text-embedding-ada-002")
            tokens = response['data'][0]['embedding']
            token_count = len(tokens)
        except OpenAIError as e:
            raise ValueError(f"Error in tokenizing text: {e}")

        # If the HTML content exceeds max_tokens, chunk it
        chunked_html = []
        if token_count > max_tokens:
            current_chunk = ""
            token_count_in_chunk = 0
            for tag in self._split_by_tags(html_content):
                current_chunk += tag
                token_count_in_chunk += len(openai.Embedding.create(input=[current_chunk], model="text-embedding-ada-002")['data'][0]['embedding'])
                
                if token_count_in_chunk > max_tokens:
                    chunked_html.append(current_chunk)
                    current_chunk = ""  # Reset chunk
                    token_count_in_chunk = 0
            if current_chunk:
                chunked_html.append(current_chunk)  # Add any remaining chunk
        else:
            chunked_html = [html_content]

        return chunked_html

    def _split_by_tags(self, html_content: str) -> List[str]:
        """
        Private method: Split the HTML content into smaller segments by finding appropriate closing tags.
        """
        segments = []
        soup = BeautifulSoup(html_content, "lxml")
        
        # Iterate over the HTML content and collect individual segments of HTML
        for element in soup.find_all(True):  # Finds all tags
            segments.append(str(element))
        
        return segments

    def _index_chunks(self, chunks: List[str], url: str) -> None:
        """
        Private method: Index each chunk into MongoDB with the associated URL.
        """
        for chunk in chunks:
            vector = self._create_vector(chunk)
            self.db.insert_one({
                "url": url,
                "content": chunk,  # Store the HTML structure chunk here
                "vector": vector
            })

    def _create_vector(self, text: str) -> List[float]:
        """
        Private method: Generate a vector for the chunk using OpenAI's embedding API.
        """
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        return response['data'][0]['embedding']

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Computes the cosine similarity between two vectors.

        Args:
            v1 (numpy.ndarray): The first vector.
            v2 (numpy.ndarray): The second vector.

        Returns:
            float: The cosine similarity score.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)