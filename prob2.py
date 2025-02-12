import telebot
import os
import chromadb
import requests
import uuid
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
from telebot.types import InputFile
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from duckduckgo_search import DDGS
from docx import Document
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Инициализация бота
TOKEN = '7563626842:AAH02HKNgRO1WwUqCZnYNWC-K7kj80uO9Kw'
bot = telebot.TeleBot(TOKEN)

# Инициализация ChromaDB
chroma_storage_path = os.path.join(os.getcwd(), "chroma_db")
client = chromadb.PersistentClient(path=chroma_storage_path)
collection = client.get_or_create_collection(
    name="chat_data",
    embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

# Функция поиска в DuckDuckGo
def duckduckgo_search_safe(query, max_results=3):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if r.get("href"):
                    results.append(r["href"])
    except Exception as e:
        logging.error(f"Ошибка DuckDuckGo: {str(e)}")
    return results

# Функция загрузки текста со страницы
def fetch_page_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs]).strip()
    except Exception as e:
        logging.error(f"Ошибка загрузки страницы {url}: {str(e)}")
        return ""

# Функция извлечения текста из PDF
def extract_text_from_pdf(file_data):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_data))
        text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])
        return text.strip()
    except Exception as e:
        logging.error(f"Ошибка извлечения текста из PDF: {str(e)}")
        return ""

# Функция извлечения текста из DOCX
def extract_text_from_docx(file_data):
    try:
        doc = Document(BytesIO(file_data))
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logging.error(f"Ошибка извлечения текста из DOCX: {str(e)}")
        return ""

# Сохранение документа в ChromaDB
def save_document_to_chromadb(user_id, text, file_name):
    try:
        if not text:
            logging.warning("Пустой текст документа. Сохранение отменено.")
            return None
        doc_id = str(uuid.uuid4())
        collection.add(ids=[doc_id], documents=[text], metadatas=[{"user_id": user_id, "file_name": file_name}])
        return doc_id
    except Exception as e:
        logging.error(f"Ошибка сохранения документа в ChromaDB: {str(e)}")
        return None

# Получение последнего загруженного документа пользователя
def get_latest_document_text(user_id):
    try:
        results = collection.get(where={"user_id": user_id}, limit=20)  # Получаем последние 20 записей
        if results and results["documents"]:
            return results["documents"][0]  # Возвращаем самый новый документ
    except Exception as e:
        logging.error(f"Ошибка получения документа из ChromaDB: {str(e)}")
    return ""

# Обработка команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    try:
        bot.send_message(message.chat.id, "🤖 Привет! Отправьте мне документ или задайте вопрос!")
    except Exception as e:
        logging.error(f"Ошибка при обработке команды /start: {str(e)}")

# Обработка загруженных документов
@bot.message_handler(content_types=['document'])
def handle_document(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        file_data = bot.download_file(file_info.file_path)
        file_name = message.document.file_name
        
        # Определяем тип файла и извлекаем текст
        if file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_data)
        elif file_name.endswith(".docx"):
            text = extract_text_from_docx(file_data)
        else:  # Предполагаем, что это текстовый файл
            text = file_data.decode("utf-8", errors="ignore")
        
        if text:
            doc_id = save_document_to_chromadb(str(message.chat.id), text, file_name)
            if doc_id:
                bot.send_message(message.chat.id, f"📂 Документ сохранен (ID: {doc_id})")
            else:
                bot.send_message(message.chat.id, "❌ Ошибка сохранения документа")
        else:
            bot.send_message(message.chat.id, "❌ Ошибка извлечения текста из документа")
    except Exception as e:
        logging.error(f"Ошибка загрузки документа: {str(e)}")
        bot.send_message(message.chat.id, f"Ошибка загрузки документа: {str(e)}")

# Обработка текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_input = message.text
        user_id = str(message.chat.id)

        # Извлекаем текст последнего загруженного документа
        context = get_latest_document_text(user_id)

        # Если документов нет, ищем в интернете
        if not context:
            urls = duckduckgo_search_safe(user_input, max_results=3)
            for url in urls:
                context += fetch_page_content(url) + "\n"

        # Если нашли информацию в документе — используем её, иначе выполняем веб-поиск
        llm = Ollama(model="llama3.2", request_timeout=60.0)
        messages = [
            ChatMessage(role="system", content="Используй документ, если он есть."),
            ChatMessage(role="user", content=f"Контекст: {context}"),
            ChatMessage(role="user", content=user_input)
        ]

        bot.send_message(user_id, "🤖 Генерирую ответ...")
        try:
            response = "".join([chunk.delta for chunk in llm.stream_chat(messages=messages)])
            bot.send_message(user_id, response)
        except Exception as e:
            logging.error(f"Ошибка генерации ответа: {str(e)}")
            bot.send_message(user_id, f"Ошибка генерации ответа: {str(e)}")
    except Exception as e:
        logging.error(f"Ошибка обработки сообщения: {str(e)}")
        bot.send_message(message.chat.id, f"Ошибка обработки сообщения: {str(e)}")

# Запуск бота
if __name__ == "__main__":
    logging.info("Бот запущен...")
    bot.polling(none_stop=True)