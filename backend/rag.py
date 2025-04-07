import os
import camelot
import fitz
from tabula import read_pdf
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from utils import logger


# Class to handle document loading and processing
class DocumentLoader:
    def __init__(self):
        self.docs = []

    # Function to load and extract data from PDF files
    def load_pdf(self, pdf_path):

        # Extract raw text using PyMuPDF
        try:
            pdf = fitz.open(pdf_path)
            text = "\n".join([page.get_text() for page in pdf])
            self.docs.append(Document(page_content=text, metadata={"source": pdf_path}))
        except Exception as e:
            logger.warning(f"[PDF TEXT FAIL] {pdf_path}: {e}")

        # Camelot tables
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
            for i, table in enumerate(tables):
                md_table = table.df.to_markdown()
                self.docs.append(
                    Document(
                        page_content=md_table,
                        metadata={"source": pdf_path, "type": "table", "table_no": i},
                    )
                )
        except Exception as e:
            logger.warning(f"[CAMELT FAIL] {pdf_path}: {e}")

        # Tabula fallback
        try:
            tabula_tables = read_pdf(pdf_path, pages="all", multiple_tables=True)
            for i, df in enumerate(tabula_tables):
                md_table = df.to_markdown()
                self.docs.append(
                    Document(
                        page_content=md_table,
                        metadata={"source": pdf_path, "type": "table", "table_no": i},
                    )
                )
        except Exception as e:
            logger.warning(f"[TABULA FAIL] {pdf_path}: {e}")

        return self.docs

    # Function to load and extract data from text files
    def load_txt(self, txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": txt_path})]
        except Exception as e:
            logger.error(f"[TXT FAIL] {txt_path}: {e}")
            return []


# Class to handle text chunking
class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Function to split documents into smaller chunks
    def chunk_text(self, documents):
        chunks = self.splitter.split_documents(documents)
        return [chunk for chunk in chunks if chunk.page_content.strip()]


# Class to handle the vector store implementation
class VectorStore:
    def __init__(self, embedding_type="ollama", model_name=None, store_path="data/faiss"):
        self.store_path = store_path
        self.embedding = self.get_embedding(embedding_type, model_name)
        self.index = self.init_faiss()

    # Function to get the appropriate embedding model
    def get_embedding(self, embedding_type, model_name):
        if embedding_type == "ollama":
            return OllamaEmbeddings(model=model_name or "mistral")
        elif embedding_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    # Function to initialize the FAISS index; If the index exists, load it; otherwise, create a new one
    def init_faiss(self):
        if os.path.exists(self.store_path):
            try:
                return FAISS.load_local(self.store_path, embeddings=self.embedding)
            except Exception as e:
                logger.warning(f"Failed to load FAISS index, creating new one: {e}")
        return None

    # Function to add documents to the vector store
    def put_vec(self, documents):
        if self.index:
            self.index.add_documents(documents)
        else:
            self.index = FAISS.from_documents(documents, embedding=self.embedding)
        self.index.save_local(self.store_path)

    # Function to perform similarity search on the vector store
    def get_vec(self, query, k=4):
        if not self.index:
            raise ValueError("Vector store not initialized. Run put_vec() first.")
        return self.index.similarity_search(query, k=k)


# Invalid entry point, should not be run directly
if __name__ == "__main__":
    exit(0)
