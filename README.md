# Digital_twin

# 🤖 AI Digital Twin (RAG-Based)

An end-to-end **RAG (Retrieval-Augmented Generation) based Digital Twin** that allows users to interact with an AI version of themselves using their resume as the knowledge base.

---

## 🚀 Overview

This project simulates a **digital twin** that can answer questions about a user by understanding their resume. Instead of giving generic answers, the system retrieves relevant information and generates responses grounded in the user’s data.

The goal was to build a system that feels like an AI version of the user — aware of their skills, education, and experience.

---

## 🧠 How It Works

1. **Resume Upload**
   - User uploads a PDF resume
   - Text is extracted using a document loader

2. **Text Processing**
   - Resume is split into smaller chunks
   - Each chunk is converted into vector embeddings

3. **Vector Database**
   - Embeddings are stored using FAISS for fast similarity search

4. **RAG Pipeline**
   - On user query:
     - Relevant chunks are retrieved
     - Passed to an LLM along with the query
     - Generates context-aware response

5. **Digital Twin Behavior**
   - Responds in **first-person ("I", "my")**
   - Only answers based on resume content
   - Rejects unrelated questions

---

## 💡 Features

- 📄 Upload and process resume (PDF)
- 🔍 Semantic search using vector embeddings
- 🤖 AI-powered responses using Groq LLM (LLaMA3)
- 🧠 Personalized responses (acts like the user)
- 💬 Interactive chat interface (Gradio)
- ⏱️ Extra utilities: time queries & basic calculations

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **FAISS (Vector Database)**
- **HuggingFace Embeddings**
- **Groq API (LLaMA3)**
- **Gradio (UI)**

---

## 📂 Project Structure
