import streamlit as st
import os
import PyPDF2
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time
import re
import sqlite3
import numpy as np
from telegram import Bot
from telegram.ext import Application

# ChromaDB
chroma_storage_path = os.path.join(os.getcwd(), "chroma_db")
chroma_settings = Settings(persist_directory=chroma_storage_path)
client = chromadb.PersistentClient(path=chroma_storage_path)

model_name = "all-MiniLM-L6-v2"
embedding_func = HuggingFaceEmbeddings(model_name=model_name)

# Collection in ChromaDB
collection_name = "chat_data"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=SentenceTransformerEmbeddingFunction(model_name=model_name)
)

# SQLite for memory storage
conn = sqlite3.connect("chat_memory.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY, user_input TEXT, response TEXT)")
conn.commit()

def save_chat(user_input, response):
    if isinstance(response, list):  
        response = " ".join(response)  # Преобразуем список в строку
    cursor.execute("INSERT INTO chat_history (user_input, response) VALUES (?, ?)", (user_input, response))
    conn.commit()


def get_chat_history():
    cursor.execute("SELECT * FROM chat_history ORDER BY id DESC LIMIT 5")
    return cursor.fetchall()

# Telegram Notification
TELEGRAM_TOKEN = "7563626842:AAH02HKNgRO1WwUqCZnYNWC-K7kj80uO9Kw"
CHAT_ID = "751123005"

async def send_telegram_notification(message):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message)

# Profanity Check
def contains_profanity(text):
    profanity_list = ["badword1", "badword2", "fuck", "shit", "bitch"]
    has_profanity = any(re.search(rf"\b{word}\b", text, re.IGNORECASE) for word in profanity_list)
    
    # Исправленный вывод (игнорируем ошибки кодировки)
    print(f"🔍 Проверяем текст: {text.encode('utf-8', errors='ignore').decode('utf-8')}")
    print(f"⚠️ Обнаружен мат: {has_profanity}")
    
    return has_profanity


def duckduckgo_search(query):
    search_results = DDGS().text(query, max_results=3)
    return "\n".join([result["body"] for result in search_results]) if search_results else "No relevant search results found."

# Process PDF Files
def process_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return text.strip()

# Process TXT Files
def process_txt(file):
    try:
        return file.read().decode("utf-8").strip()
    except Exception as e:
        st.error(f"Error processing TXT: {e}")
        return ""

# Streamlit UI
st.title("Final Project: AI-Powered Knowledge Base")
menu = st.radio("Select an action", ["Add data", "Show data", "Chat Bot"])

if menu == "Add data":
    st.subheader("Adding data to the database")
    input_text = st.text_area("Enter text")
    file = st.file_uploader("Or upload PDF/TXT file", type=["pdf", "txt"])
    
    if st.button("Save data"):
        text = ""
        if input_text.strip():
            text = input_text
        elif file is not None:
            text = process_pdf(file) if file.type == "application/pdf" else process_txt(file)
        
        if not text:
            st.error("File is empty or text is missing! Upload a valid file or enter text.")
        elif contains_profanity(text):
            st.error("Profanity detected! Please remove inappropriate content.")
        else:
            embeddings = SentenceTransformer(model_name).encode([text]).tolist()
            doc_id = f"doc_{len(collection.get().get('ids', [])) + 1}"
            collection.add(documents=[text], embeddings=embeddings, ids=[doc_id])
            st.success(f"Added document: {doc_id}")

if menu == "Show data":
    st.subheader("Saved documents in ChromaDB")
    docs = collection.get().get("documents", [])
    if docs:
        for i, doc in enumerate(docs, start=1):
            st.write(f"{i}. {doc[:500]}...")
    else:
        st.write("No saved data!")


if menu == "Chat Bot":
    st.subheader("Chat Bot (ChromaDB + Ollama + DuckDuckGo)")
    user_input = st.text_input("Enter your question:")

    if st.button("Submit"):
        if user_input.strip():
            response_placeholder = st.empty()
            response = ""

            # 1️⃣ Поиск в ChromaDB
            query_embedding = SentenceTransformer(model_name).encode([user_input]).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=1)
            if results and results.get("documents"):
                chroma_response = results["documents"][0][0]  # Берём первую строку из первого документа
                chroma_response = chroma_response.strip()  # Убираем лишние пробелы и символы
            else:
                chroma_response = "⚠️ No relevant data found in ChromaDB."



            # 2️⃣ Генерация ответа через Ollama
            chat_history = get_chat_history()
            history_text = "\n".join([f"User: {h[1]}\nBot: {h[2]}" for h in chat_history])

            messages = [
                ChatMessage(role="user", content=history_text),
                ChatMessage(role="user", content=user_input)
            ]

            llm = Ollama(model="llama3.2", request_timeout=120.0)

            try:
                response_stream = llm.stream_chat(messages=messages)
                for chunk in response_stream:
                    response += chunk.delta
                    response_placeholder.write(response)
            except Exception as e:
                response = f"Error generating response: {str(e)}"

            # 3️⃣ Дополнительная информация из DuckDuckGo
            search_results = duckduckgo_search(user_input)
            response += f"\n\n🌍 Additional info from DuckDuckGo:\n{search_results}"

            # 4️⃣ Объединяем ответ ChromaDB + Ollama + DuckDuckGo
            if chroma_response and "⚠️" not in chroma_response:
                response = f"📚 ChromaDB result:\n{chroma_response}\n\n🤖 AI Answer:\n{response}"


            # 5️⃣ Сохранение в память
            save_chat(user_input, response)

            # 6️⃣ Уведомление в Telegram
            import asyncio
            asyncio.run(send_telegram_notification(f"New query processed: {user_input}"))

            st.write(response)
        else:
            st.warning("Please enter a valid question.")
