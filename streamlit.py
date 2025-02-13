import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import os
import PyPDF2
import chardet
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sqlite3
from datetime import datetime
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import time
import requests
# ----------------- SQLite Database Setup -----------------
conn = sqlite3.connect("documents.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT,
        filename TEXT,
        file_type TEXT,
        content_length INTEGER,
        timestamp TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_name TEXT UNIQUE
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        role TEXT,
        content TEXT,
        timestamp TEXT,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )
""")
conn.commit()

def save_metadata(doc_id, filename, file_type, content_length):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(
        "INSERT INTO documents (doc_id, filename, file_type, content_length, timestamp) VALUES (?, ?, ?, ?, ?)",
        (doc_id, filename, file_type, content_length, timestamp)
    )
    conn.commit()

# ChromaDB and Embedding Model Setup 
chroma_storage_path = os.path.join(os.getcwd(), "chroma_db")
chroma_settings = Settings(persist_directory=chroma_storage_path)
client = chromadb.Client(chroma_settings)

model_name = "all-MiniLM-L6-v2"
embedded_func = SentenceTransformerEmbeddingFunction(model_name=model_name)
collection_name = "knowledge_base"
collection = client.get_or_create_collection(name=collection_name, embedding_function=embedded_func)

def duckduckgo_search(query, max_results=3):
    results = []# Initialize empty list to store URLs
    with DDGS() as ddgs:  # Create a DuckDuckGo search session
        try:
            for r in ddgs.text(query, max_results=max_results):  # Perform search query and retrieve results
                if r.get("href"): # Extract and store only valid URLs
                    results.append(r["href"])
                time.sleep(1)  # Delay to prevent rate-limiting
        except Exception as e:
            print(f"Error during search: {str(e)}") # Print error if search fails
    return results

# Function to fetch and extract text content from a given webpage
def fetch_page_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"} # Set user-agent to avoid bot detection
        resp = requests.get(url, headers=headers, timeout=10) # Send GET request to fetch the page
        resp.raise_for_status() # Raise an error if the request fails
        soup = BeautifulSoup(resp.text, "html.parser")  # Parse the webpage content using BeautifulSoup
        paragraphs = soup.find_all("p")  # Extract all paragraph elements
        page_text = " ".join([p.get_text() for p in paragraphs])  # Combine text from all paragraphs
        return page_text
    except Exception:
        return ""
# ------------------ TEXT PROCESSING ------------------
# Function to truncate text to a specified word limit
def truncate_text(text, max_words=500):
    words = text.split() 
    if len(words) <= max_words:
        return text
    truncated_text = " ".join(words[:max_words]) # Keep only the first 500 words
    last_period_index = truncated_text.rfind(".")  # Find the last full sentence
    if last_period_index != -1:
        truncated_text = truncated_text[:last_period_index + 1]  # Keep full sentence
    return truncated_text


st.title("AI-Powered Chatbot") 

menu = st.radio("Select an action: ", [
    "Add data",
    "Show data",
    "Chat with AI",
    "Visualize Insights"
])

def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f" Error processing PDF: {e}")
        return None

def process_txt(file):
    try:
        raw_data = file.read()
        encoding = chardet.detect(raw_data)["encoding"]
        return raw_data.decode(encoding)
    except Exception as e:
        st.error(f" Error processing TXT: {e}")
        return None

def save_to_chroma(content):
    embedding = SentenceTransformer(model_name).encode(content).tolist()
    doc_id = f"doc_{len(collection.get()['ids']) + 1}"
    collection.add(documents=[content], embeddings=[embedding], ids=[doc_id])
    return doc_id

# ----------------- Application Menu Options -----------------
if menu == "Add data":
    st.subheader(" Upload PDF or TXT File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            content = process_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            content = process_txt(uploaded_file)
        else:
            content = None

        if content:
            doc_id = save_to_chroma(content)
            save_metadata(doc_id, uploaded_file.name, uploaded_file.type, len(content))
            st.success(f"Document saved as: {doc_id}")
        else:
            st.error(" Could not extract content from the file!")

elif menu == "Show data":
    st.subheader(" Stored Documents (ChromaDB)")
    documents = collection.get()["documents"]
    st.write(documents if documents else "No documents stored yet.")

elif menu == "Chat with AI":
    st.subheader("Chat with the Knowledge Base AI")
    cursor.execute("SELECT id, chat_name FROM chats")
    chats = cursor.fetchall()
    chat_dict = {name: chat_id for chat_id, name in chats}
    use_web = st.checkbox("Search the web", value=False)  # Опция поиска в интернете
    chat_name = st.text_input("Enter a new chat topic or select one below:")
    
    if chat_name and chat_name not in chat_dict:
        cursor.execute("INSERT INTO chats (chat_name) VALUES (?)", (chat_name,))
        conn.commit()
        st.rerun()

    selected_chat = st.selectbox("Select a chat topic:", list(chat_dict.keys()))
    chat_id = chat_dict.get(selected_chat)

    cursor.execute("SELECT role, content FROM messages WHERE chat_id = ?", (chat_id,))
    chat_history = cursor.fetchall()
    
    for role, content in chat_history:
        st.write(f"{role.capitalize()}: {content}")

    user_input = st.text_input("Enter your message:")
    if st.button("Send") and user_input:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                       (chat_id, "user", user_input, timestamp))
        conn.commit()
        
        # Если выбран веб-поиск
        if use_web:
            st.write("Searching the web...")
            search_results = duckduckgo_search(user_input, max_results=3)  # Поиск в интернете
            web_content = ""
            for url in search_results:
                content = fetch_page_content(url)  # Извлечение текста с каждой страницы
                web_content += truncate_text(content) + "\n\n"  # Обрезка текста и добавление в контекст
            if web_content:
                user_input += "\n\nWeb search results:\n" + web_content  # Добавление результатов в запрос

        # Векторизация запроса
        query_vector = SentenceTransformer(model_name).encode(user_input).tolist()
        results = collection.query(query_embeddings=query_vector, n_results=3)
        
        # Подготовка сообщений для модели
        messages = [ChatMessage(role="user", content=user_input)]
        if results["documents"]:
            messages.append(ChatMessage(role="user", content="Considering relevant knowledge base documents."))

        # Генерация ответа с помощью модели
        llm = Ollama(model="llama3.2", request_timeout=90.0)
        response = "".join(chunk.delta for chunk in llm.stream_chat(messages=messages))
        
        # Сохранение ответа в базу данных
        cursor.execute("INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                       (chat_id, "assistant", response, timestamp))
        conn.commit()
        st.rerun()

elif menu == "Visualize Insights":
    st.subheader(" Knowledge Base Insights")
    documents = collection.get()["documents"]
    all_text = " ".join(documents) if documents else "No data available."
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    st.image(wordcloud.to_array(), use_container_width=True)
