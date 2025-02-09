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

# AUTHOR: Zhanel77
