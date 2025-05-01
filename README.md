# Excel Q&A Chatbot

A chatbot that answers user questions based on an uploaded Excel sheet (Q&A pairs). Powered by FAISS and Mistral LLM via Ollama.

## Features

- Upload Excel with 2 columns: `Question`, `Answer`
- Semantic search with embeddings (SentenceTransformers)
- Local LLM response (Mistral via Ollama)
- Simple Streamlit frontend

## Run Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start Ollama with Mistral:
   ```
   ollama run mistral
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

## Deployment (Optional)

To host publicly for free, push this repo to GitHub and deploy via [Streamlit Cloud](https://streamlit.io/cloud).
