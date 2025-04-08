import os
from dotenv import load_dotenv
import requests
from dataclasses import dataclass
from typing import List
from langchain.schema import Document

load_dotenv()


# Data class for search results, formatted for LangChain
@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str


# Search class to handle GCSE interactions and data extraction
class Search:
    def __init__(self):
        self.gcse_key = os.getenv("GCSE_API_KEY")
        self.gcse_cx = os.getenv("GCSE_CX")
        if not self.gcse_key or not self.gcse_cx:
            raise ValueError("Missing GCSE_API_KEY or GCSE_CX in environment variables.")

    def search(self, query: str) -> dict:
        params = {
            "key": self.gcse_key,
            "cx": self.gcse_cx,
            "q": query,
        }
        try:
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Error performing search: {e}")

    def extract_info(self, results: dict, top_k: int = 3) -> List[SearchResult]:
        extracts = []
        for item in results.get("items", [])[:top_k]:
            extracts.append(
                SearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                )
            )
        return extracts

    def to_documents(self, extracts: List[SearchResult]) -> List[Document]:
        return [
            Document(
                page_content=extract.snippet,
                metadata={"title": extract.title, "link": extract.link},
            )
            for extract in extracts
        ]


# Example usage, to be removed later
if __name__ == "__main__":
    search_instance = Search()
    query = "Python programming"
    try:
        results = search_instance.search(query)
        extracts = search_instance.extract_info(results, top_k=5)

        for extract in extracts:
            print(f"\nTitle: {extract.title}\nLink: {extract.link}\nSnippet: {extract.snippet}\n")

        docs = search_instance.to_documents(extracts)
        print(f"\nGenerated {len(docs)} LangChain Document objects.")

    except Exception as e:
        print(f"An error occurred: {e}")
