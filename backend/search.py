import os
from dotenv import load_dotenv
import requests


class Search:
    def __init__(self):
        load_dotenv()
        self.gcse_key = os.getenv("GCSE_API_KEY")
        self.gcse_cx = os.getenv("GCSE_CX")

    # Function to perform a search using the Google Custom Search API
    def search(self, query):
        params = f"key={self.gcse_key}&cx={self.gcse_cx}&q={query}"
        url = f"https://www.googleapis.com/customsearch/v1?{params}"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
    # Function to extract relevant information from the search results
    def extract_info(self, results):
        extracts = []
        for item in results.get("items", []):
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet")
            extracts.append({"title": title, "link": link, "snippet": snippet})
        return extracts
    

# Example usage, to be removed later
if __name__ == "__main__":
    search_instance = Search()
    query = "Python programming"
    try:
        results = search_instance.search(query)
        with open("search_results.json", "w") as f:
            f.write(str(results))
        extracts = search_instance.extract_info(results)
        for extract in extracts:
            print(extract)
    except Exception as e:
        print(f"An error occurred: {e}")