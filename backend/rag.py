import os
import pickle
import camelot
import fitz
from tabula import read_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from utils import logger


# ------------------------------------------------------------------------
class DocumentLoader:
    def __init__(self):
        pass

    def load_pdf(self, file_path):
        pass

    def load_txt(self, file_path):
        pass

    def load_md(self, file_path):
        pass


class Chunker:
    def __init__(self):
        pass

    def chunk_text(self, text):
        pass

    def embed_chunk(self, chunk):
        pass


class VectorStore:
    def __init__(self):
        pass

    def put_vec(self, text):
        pass

    def get_vec(self, query):
        pass


# ------------------------------------------------------------------------


# Invalid entry point, should not be run directly
if __name__ == "__main__":
    exit(0)
