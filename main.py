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

# Проверяем, загружена ли модель SentenceTransformer
model_name = "all-MiniLM-L6-v2"
try:
    embedding_model = SentenceTransformer(model_name)
    print("Модель SentenceTransformer загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки модели: {str(e)}")

# Инициализация бота с вашим токеном
bot = telebot.TeleBot('7563626842:AAH02HKNgRO1WwUqCZnYNWC-K7kj80uO9Kw')

# ChromaDB
chroma_storage_path = os.path.join(os.getcwd(), "chroma_db")
chroma_settings = Settings(persist_directory=chroma_storage_path)
client = chromadb.PersistentClient(path=chroma_storage_path)

embedding_func = HuggingFaceEmbedding(model_name=model_name)  

# Collection in ChromaDB
collection_name = "chat_data"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=SentenceTransformerEmbeddingFunction(model_name=model_name)
)

# Безопасный поиск DuckDuckGo с обработкой ошибок
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
            print(f"Ошибка DuckDuckGo (попытка {attempt+1}): {str(e)}")
            time.sleep(5)
    return []

# Функция для получения текста со страницы
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

# Ограничение количества слов в тексте
def truncate_text(text, max_words=500):
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated_text = " ".join(words[:max_words])
    last_period_index = truncated_text.rfind(".")
    if last_period_index != -1:
        truncated_text = truncated_text[:last_period_index + 1]
    return truncated_text

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я бот, который использует Ollama и DuckDuckGo. Задайте мне вопрос, и я постараюсь ответить!")

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text

    # Поиск информации через DuckDuckGo
    urls = duckduckgo_search_safe(user_input, max_results=3)
    bot.send_message(message.chat.id, f"🔎 Найдено {len(urls)} источников, извлекаю текст...")

    context = ""
    for url in urls:
        page_text = fetch_page_content(url)
        truncated_text = truncate_text(page_text, max_words=500)
        if len(truncated_text.strip()) > 30:
            context += f"\n🔗 Источник: {url}\n{truncated_text}\n"

    # Формируем запрос для Ollama
    messages = [ChatMessage(role="user", content=context)] if context else []
    messages.append(ChatMessage(role="user", content=user_input))

    # Инициализация Ollama
    llm = Ollama(model="llama3.2", request_timeout=120.0)
    bot.send_message(message.chat.id, "🤖 Генерирую ответ...")

    try:
        response_stream = llm.stream_chat(messages=messages)
        response = ""
        for chunk in response_stream:
            response += chunk.delta
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Ошибка генерации ответа: {str(e)}")

# Запуск бота
bot.polling(none_stop=True)
