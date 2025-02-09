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
import uuid  # ✅ Генерация уникального ID

# ✅ Инициализация бота
TOKEN = '7563626842:AAH02HKNgRO1WwUqCZnYNWC-K7kj80uO9Kw'
bot = telebot.TeleBot(TOKEN)

# ✅ Настройки ChromaDB
chroma_storage_path = os.path.join(os.getcwd(), "chroma_db")
client = chromadb.PersistentClient(path=chroma_storage_path)

# ✅ Создание коллекции ChromaDB
collection_name = "chat_data"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

# ✅ Функция безопасного поиска DuckDuckGo
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
            print(f"❌ Ошибка DuckDuckGo (попытка {attempt+1}): {str(e)}")
            time.sleep(5)
    return []

# ✅ Получение текста со страницы
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

# ✅ Ограничение количества слов в тексте
def truncate_text(text, max_words=500):
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated_text = " ".join(words[:max_words])
    last_period_index = truncated_text.rfind(".")
    if last_period_index != -1:
        truncated_text = truncated_text[:last_period_index + 1]
    return truncated_text

# ✅ Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(
        message.chat.id,
        "👋 Привет! Я бот, который использует Ollama и DuckDuckGo. Задайте мне вопрос, и я постараюсь ответить!\n"
        "📌 Доступные команды:\n"
        "🔹 /history - Посмотреть последние запросы\n"
        "🔹 Отправьте фото или документ, и я его сохраню."
    )

# ✅ Обработчик команды /history (показ последних 5 записей из ChromaDB)
@bot.message_handler(commands=['history'])
def show_history(message):
    try:
        results = collection.get()
        if "documents" not in results or len(results["documents"]) == 0:
            bot.send_message(message.chat.id, "📭 История пуста.")
            return

        history_messages = []
        for i, doc in enumerate(results["documents"][-5:]):  # Берем последние 5 записей
            history_messages.append(f"🔹 {i+1}: {doc[:300]}...")  # Обрезаем текст для читаемости
        
        bot.send_message(message.chat.id, "📜 Последние запросы:\n\n" + "\n\n".join(history_messages))

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка при получении истории: {str(e)}")

# ✅ Обработчик документов
@bot.message_handler(content_types=['document'])
def handle_document(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        save_path = os.path.join("downloads", message.document.file_name)
        os.makedirs("downloads", exist_ok=True)

        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.send_message(message.chat.id, f"📄 Документ '{message.document.file_name}' сохранен!")
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка при сохранении документа: {str(e)}")

# ✅ Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    user_id = str(message.chat.id)

    # ✅ Поиск информации через DuckDuckGo
    urls = duckduckgo_search_safe(user_input, max_results=3)
    bot.send_message(message.chat.id, f"🔎 Найдено {len(urls)} источников, извлекаю текст...")

    context = ""
    for url in urls:
        page_text = fetch_page_content(url)
        truncated_text = truncate_text(page_text, max_words=500)
        if len(truncated_text.strip()) > 30:
            context += f"\n🔗 Источник: {url}\n{truncated_text}\n"

    # ✅ Формирование запроса для Ollama
    messages = []
    if context:
        messages.append(ChatMessage(role="user", content=f"Контекст: {context}"))

    messages.append(ChatMessage(role="user", content=user_input))

    # ✅ Используем Ollama
    llm = Ollama(model="llama3.2", request_timeout=120.0)
    bot.send_message(message.chat.id, "🤖 Генерирую ответ...")

    try:
        response_stream = llm.stream_chat(messages=messages)
        response = ""
        for chunk in response_stream:
            response += chunk.delta

        bot.send_message(message.chat.id, response)

        # ✅ Сохранение данных в ChromaDB
        doc_id = str(uuid.uuid4())  # Генерация уникального ID
        collection.add(
            ids=[doc_id],
            documents=[f"🔹 Запрос: {user_input}\n🔹 Контекст: {context}\n🔹 Ответ: {response}"],
            metadatas=[{"user_id": user_id, "query": user_input}]
        )
        print(f"✅ Данные сохранены в ChromaDB (ID: {doc_id})")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка генерации ответа: {str(e)}")

# ✅ Запуск бота
bot.polling(none_stop=True)
