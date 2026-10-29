# 📚 AskMyNotes

Your Personalized Study Assistant: chat with your notes, quiz yourself, and get summaries, powered by Retrieval-Augmented Generation (RAG).

## Features
- Chat with your PDF notes and get answers with source citations (file and page)
- Follow-up questions understood in context
- Quiz mode: 5 multiple-choice questions generated from your notes
- Summary mode: key points and likely exam topics
- Upload, delete and filter PDFs from the sidebar
- Light and dark theme

## How it works
PDF → text chunks → embeddings (sentence-transformers) → Chroma vector database → retrieval of the most relevant chunks → LLM (Groq) → answer with citations

## Tech stack
Python, Streamlit, ChromaDB, sentence-transformers, pypdf, Groq API

## Setup
1. Clone the repo and open the folder
2. Create and activate a virtual environment:
   `python -m venv venv` then `venv\Scripts\activate`
3. Install packages: `pip install -r requirements.txt`
4. Create a `.env` file with your key: `GROQ_API_KEY=your-key-here`
5. Run: `streamlit run app.py`
6. Upload PDFs in the sidebar and click **Rebuild index**

## Limitations
- Works with text-based PDFs only (scanned images need OCR)
- Depends on the free-tier limits of the LLM API

## Screenshots
![App screenshot](docs/screenshot.png)