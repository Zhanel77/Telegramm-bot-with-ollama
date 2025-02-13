# Telegram Bot with Ollama & ChromaDB

This is an advanced Telegram bot that integrates Ollama for AI-powered responses, DuckDuckGo for web searches, and ChromaDB for storing user queries and generated answers.

## Features

- AI-powered responses using **Ollama** (Llama3.2 model)
- Web search integration via **DuckDuckGo**
- Query storage using **ChromaDB**
- Retrieve recent queries with the `/history` command
- Supports document and image uploads

## Installation

### Prerequisites

Make sure you have the following installed:

- **Python 3.10+**
- **pip**
- **Git**
- **Virtual environment (optional but recommended)**

### Steps to Install

```sh
# Clone the repository
git clone https://github.com/Zhanel77/Telegramm-bot-with-ollama.git
cd Telegramm-bot-with-ollama

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the bot using:

```sh
python bot.py
```

## Available Commands

- `/start` - Start the bot
- `/history` - Show last 5 stored queries
- **Send a text message** - The bot will generate an AI-powered response
- **Send a document** - The bot will save the document
- **Send an image** - The bot will store the image

## Technologies Used

- **Python** 
- **Telebot** (Telegram API)
- **Ollama (Llama3.2)** for AI-powered responses
- **DuckDuckGo Search API** for retrieving information
- **ChromaDB** for storing and retrieving chat history
- **BeautifulSoup** for extracting webpage text
- **Sentence Transformers** for text embeddings


# Development
If you want to improve the bot:
```sh
git checkout -b feature-branch
```
# Make changes
```sh
git commit -m "Added a new feature"
git push origin feature-branch
```


# AI-Powered Chatbot

This project is an AI chatbot that allows users to:

- Upload and process files (PDF, TXT)
- Store text data in SQLite and ChromaDB
- Interact with AI using Llama and ChromaDB
- Search for information on the web using DuckDuckGo
- Analyze and visualize data with a word cloud

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-repository/ai-chatbot.git
cd ai-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

## 🛠 Features

### 1️⃣ Add Data ("Add data")

- Upload PDF and TXT files
- Extract text
- Store text in ChromaDB
- Save metadata in SQLite

### 2️⃣ View Data ("Show data")

- Display uploaded documents from ChromaDB

### 3️⃣ Chat with AI ("Chat with AI")

- Create and manage chats
- Search for information in the knowledge base (ChromaDB)
- Generate responses using the Llama model
- Perform web searches when needed

### 4️⃣ Data Analysis ("Visualize Insights")

- Generate a word cloud from uploaded documents

## 📂 Project Structure

```
├── app.py                # Main application code
├── requirements.txt      # Dependency list
├── README.md             # Project documentation
├── documents.db          # SQLite database (auto-generated)
├── chroma_db/            # Vector storage (ChromaDB)
```

## 🛠 Technologies Used

- **Streamlit** - User interface
- **ChromaDB** - Vector storage
- **SQLite** - Database
- **SentenceTransformer** - Text processing
- **LlamaIndex** - AI-powered responses
- **DuckDuckGo Search** - Web search
- **WordCloud** - Data visualization

## 🔥 Developers

Kuandyk Zhanel, Seipolla Koblandy, Kazbekova Zhaniya

📌 **Note**: To run the AI chatbot, make sure Ollama is installed and the `llama3.2` model is downloaded.


