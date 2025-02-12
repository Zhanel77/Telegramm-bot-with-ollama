import streamlit as st
import chromadb
import os
import hashlib
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from chromadb.errors import UniqueConstraintError
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS
import asyncio
import httpx
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from collections import Counter
import pandas as pd

# ---------------------------
# ChromaDB and Embedding Setup
# ---------------------------
DB_DIRECTORY = os.path.join(os.getcwd(), "search_db")
os.makedirs(DB_DIRECTORY, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_DIRECTORY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class ChromaDBEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.embedding_model.encode(input).tolist()

embedding = ChromaDBEmbeddingFunction(embedding_model)

try:
    collection = chroma_client.create_collection(
        name="rag_collection",
        metadata={"description": "RAG Collection with Ollama"},
        embedding_function=embedding
    )
except UniqueConstraintError:
    collection = chroma_client.get_collection("rag_collection")

MODEL_NAME = "llama3.2"

# ---------------------------
# Helper Functions
# ---------------------------
def compute_hash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def add_documents(documents, ids):
    collection.add(documents=documents, ids=ids)

def query_documents(query_text, n_results=None):
    if n_results is None:
        results = collection.get()
        return results["documents"]
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"]

def ollama_generate(prompt):
    llm = OllamaLLM(model=MODEL_NAME)
    return llm.invoke(prompt)

async def scrape_page(url):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

async def scrape_pages(urls):
    tasks = [scrape_page(url) for url in urls]
    return await asyncio.gather(*tasks)

def extract_keywords(query, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([query])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf.toarray()[0]
    top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]

@st.cache_data
def search_duckduckgo(query, max_results=5):
    try:
        results = DDGS().text(query, max_results=max_results)
        urls = [result['href'] for result in results if 'href' in result]
        contents = asyncio.run(scrape_pages(urls))

        content_list = []
        metadata_list = []

        for i, content in enumerate(contents):
            metadata_list.append({"source": urls[i], "content": content})
            content_list.append(content)
            collection.add(documents=[content], metadatas=[{"url": urls[i]}], ids=[f"{hash(content)}"])

        return content_list, metadata_list
    except Exception as e:
        return [], []

def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
    else:
        text = None
    return text

def upload_file():
    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
        try:
            text_content = process_uploaded_file(uploaded_file)
            if text_content:
                doc_hash = compute_hash(text_content)
                existing_docs = collection.get()
                existing_hashes = [compute_hash(doc) for doc in existing_docs['documents']]

                if doc_hash in existing_hashes:
                    st.warning("This document is already in the database.")
                else:
                    doc_id = f"doc{len(existing_docs['documents']) + 1}"
                    embeddings = embedding([text_content])[0]
                    collection.add(
                        ids=[doc_id],
                        documents=[text_content],
                        embeddings=[embeddings]
                    )
                    st.success(f"File processed and saved with ID: {doc_id}")
            else:
                st.error("Unsupported file type or empty file.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------------------
# Main Streamlit App
# ---------------------------
def main():
    st.title("Smart AI Chatbot")

    menu = ["Home", "View Documents", "Add Document", "Ask Ollama"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.subheader("Welcome to the Intelligent Knowledge Hub!")
        st.text("📌 Use the sidebar to navigate and interact with the app.")

    elif choice == "View Documents":
        if st.button("Delete all documents"):
            try:
                collection.delete(where={"id": {"$ne": ""}})
                st.sidebar.success("All documents deleted successfully.")
            except Exception as e:
                st.sidebar.error(f"Error deleting documents: {e}")
        docs = collection.get()
        if docs['documents']:
            for i, (doc, doc_id) in enumerate(zip(docs['documents'], docs['ids'])):
                with st.expander(f"Document {i + 1} (ID: {doc_id})"):
                    st.write(doc)
                    if st.button(f"Hide Document {i + 1}", key=f"hide_{doc_id}"):
                        collection.delete(ids=[doc_id])
                        st.success(f"Document {i + 1} (ID: {doc_id}) deleted.")
        else:
            st.warning("No documents found.")

    elif choice == "Add Document":
        upload_file()

    elif choice == "Ask Ollama":
        user_input = st.text_input("Enter your query:")
        if user_input:
            with st.spinner("Processing your query..."):
                response = ollama_generate(user_input)
                st.write("**Response:**")
                st.write(response)

if __name__ == "__main__":
    main()