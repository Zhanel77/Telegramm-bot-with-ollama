import telebot
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from duckduckgo_search import DDGS
import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from bs4 import BeautifulSoup
import time
import uuid  

# Initialize the bot
TOKEN = '7563626842:AAH02HKNgRO1WwUqCZnYNWC-K7kj80uO9Kw'
bot = telebot.TeleBot(TOKEN)

# ChromaDB settings
chroma_storage_path = os.path.join(os.getcwd(), "chroma_db")
client = chromadb.PersistentClient(path=chroma_storage_path)

# Create a ChromaDB collection
collection_name = "chat_data"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

# Safe DuckDuckGo search function
def duckduckgo_search_safe(query, max_results=3, retries=3):
    results = []
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    if r.get("href"):
                        results.append(r["href"])
            return results
        except Exception as e:
            print(f"DuckDuckGo error (attempt {attempt+1}): {str(e)}")
            time.sleep(5)
    return []

# Retrieve text from a webpage
def fetch_page_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        page_text = " ".join([p.get_text() for p in paragraphs])
        return page_text.strip()
    except Exception:
        return ""

# Limit the number of words in text
def truncate_text(text, max_words=500):
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated_text = " ".join(words[:max_words])
    last_period_index = truncated_text.rfind(".")
    if last_period_index != -1:
        truncated_text = truncated_text[:last_period_index + 1]
    return truncated_text

# Handler for the /start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(
        message.chat.id,
        " 👋 Hello! I am a bot that uses Ollama and DuckDuckGo. Ask me a question, and I will try to answer!\n"
        " 🔗 Available commands:\n"
        " 📍 /history - View recent queries\n"
        " 📷 Send a photo or document, and I will save it."
    )

# Handler for the /history command (show last 5 records from ChromaDB)
@bot.message_handler(commands=['history'])
def show_history(message):
    try:
        results = collection.get()
        if "documents" not in results or len(results["documents"]) == 0:
            bot.send_message(message.chat.id, "📭 History is empty.")
            return

        history_messages = []
        for i, doc in enumerate(results["documents"][-5:]):  # Get the last 5 records
            history_messages.append(f"🔹 {i+1}: {doc[:500]}...")  # Trim text for readability
        
        bot.send_message(message.chat.id, "Recent queries:" + " ".join(history_messages))

    except Exception as e:
        bot.send_message(message.chat.id, f"Error retrieving history: {str(e)}")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)  # We get photos in maximum quality
        downloaded_file = bot.download_file(file_info.file_path)
        
        save_path = os.path.join("downloads", f"photo_{message.chat.id}_{int(time.time())}.jpg")
        os.makedirs("downloads", exist_ok=True)

        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.send_message(message.chat.id, f"📷 Photo saved as '{save_path}'!")
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error saving photo: {str(e)}")

# Handler for document uploads
@bot.message_handler(content_types=['document'])
def handle_document(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        save_path = os.path.join("downloads", message.document.file_name)
        os.makedirs("downloads", exist_ok=True)

        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.send_message(message.chat.id, f"Document '{message.document.file_name}' saved!")
    except Exception as e:
        bot.send_message(message.chat.id, f"Error saving document: {str(e)}")

# Handler for text messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    user_id = str(message.chat.id)

    # Search for information using DuckDuckGo
    urls = duckduckgo_search_safe(user_input, max_results=3)
    bot.send_message(message.chat.id, f"🔍 Found {len(urls)} sources, extracting text...")

    context = ""
    for url in urls:
        page_text = fetch_page_content(url)
        truncated_text = truncate_text(page_text, max_words=500)
        if len(truncated_text.strip()) > 30:
            context += f"\n Source: {url}\n{truncated_text}\n"

    # Prepare query for Ollama
    messages = []
    if context:
        messages.append(ChatMessage(role="user", content=f"Context: {context}"))

    messages.append(ChatMessage(role="user", content=user_input))

    # Use Ollama
    llm = Ollama(model="llama3.2", request_timeout=90.0)
    bot.send_message(message.chat.id, " 🤖 Generating response...")

    try:
        response_stream = llm.stream_chat(messages=messages)
        response = ""
        for chunk in response_stream:
            response += chunk.delta

        bot.send_message(message.chat.id, response)

        # Save data to ChromaDB
        doc_id = str(uuid.uuid4())  # Generate a unique ID
        collection.add(
            ids=[doc_id],
            documents=[f"🔹 Query: {user_input}\n🔹 Context: {context}\n🔹 Response: {response}"],
            metadatas=[{"user_id": user_id, "query": user_input}]
        )
        print(f"✅ Data saved to ChromaDB (ID: {doc_id})")

    except Exception as e:
        bot.send_message(message.chat.id, f"Error generating response: {str(e)}")

# Start the bot
bot.polling(none_stop=True)
