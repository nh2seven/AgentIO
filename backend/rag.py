import os
import pickle
import camelot
import fitz
from tabula import read_pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter


# TfidfRetriever class
class Retriever:
    def __init__(self, vector_store, search_kwargs):
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs

    # Return the top k relevant documents for a given query
    def get_docs(self, query):
        k = self.search_kwargs.get("k", 4)
        return self.vector_store.similarity_search(query, k=k)


# Document class to mimic LangChain Document structure
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


# Custom TF-IDF Vector Store
class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents = []
        self.tfidf_matrix = None

    # Add documents to the vector store
    def add_docs(self, documents):
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self

    # Save the vector store to disk
    def vec_save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)

        with open(f"{dir_path}/vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(f"{dir_path}/tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(f"{dir_path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    # Load the vector store from disk
    def vec_load(cls, dir_path):
        vector_store = cls()

        with open(f"{dir_path}/vectorizer.pkl", "rb") as f:
            vector_store.vectorizer = pickle.load(f)
        with open(f"{dir_path}/tfidf_matrix.pkl", "rb") as f:
            vector_store.tfidf_matrix = pickle.load(f)
        with open(f"{dir_path}/documents.pkl", "rb") as f:
            vector_store.documents = pickle.load(f)

        return vector_store

    # Perform similarity search to find the most relevant documents
    def similarity(self, query, k=4):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]

        return [self.documents[i] for i in top_indices]

    # Return a retriever object compatible with LangChain
    def as_retriever(self, search_kwargs=None):
        search_kwargs = search_kwargs or {}
        return Retriever(self, search_kwargs)
