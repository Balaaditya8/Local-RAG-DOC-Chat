# 📄 LOCAL-RAG-DOC-Chat

This is a **Retrieval-Augmented Generation (RAG)** application that allows you to **upload a PDF** and chat with it using a **local LLM (Mistral via Ollama)**. The app processes the PDF, stores its content in a **ChromaDB vector store**, and uses an embedding-based retrieval mechanism to generate accurate responses.

## 🚀 Features

- 📂 **Upload a PDF** and extract its text
- 🧠 **Chunk and store text** as embeddings in ChromaDB
- 🔍 **Retrieve relevant chunks** for answering user queries
- 💡 **Use Mistral via Ollama** for local LLM-based responses
- 💬 **Interactive chat interface** with history retention

## 🛠️ Installation

Ensure you have **Python 3.9+** and `pip` installed.  

### 1 Clone the Repository

```bash
git clone https://github.com/Balaaditya8/RAG-DOC-Chat-Ollama.git
cd RAG-DOC-Chat-Ollama
```

### 2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3 Install Ollama and Mistral Model

[Ollama](https://ollama.com) is required to run the local LLM.

1. **Install Ollama**:
   - Mac: `brew install ollama`
   - OR -  Follow [Ollama installation guide](https://ollama.com)

2. **Download Mistral Model**:
   ```bash
   ollama pull mistral
   ```

## ▶️ Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

## 📌 Usage

1. **Upload a PDF**
2. Wait for processing (converts PDF text into embeddings)
3. Ask questions in the chat interface
4. The app retrieves relevant information and generates answers

## 🛋️ Dependencies

The `requirements.txt` includes:

```txt
streamlit
langchain
langchain_community
langchain_chroma
langchain_core
sentence-transformers
chromadb
pypdf
ollama
```


