# Excel Q&A Chatbot

A Streamlit app that turns a spreadsheet of Q&A pairs into a chatbot. Upload an
Excel file or paste a public Google Sheet URL, then ask questions in plain
language and get answers grounded in your rows.

[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B)](https://streamlit.io/cloud)

## How it works

1. **Embed.** Every question in the sheet is encoded once with
   `sentence-transformers` (`all-MiniLM-L6-v2`) and indexed in FAISS.
2. **Retrieve.** Your question is encoded the same way; FAISS returns the
   nearest Q&A rows by cosine similarity.
3. **Generate.** Those rows are handed to Cohere `command-r+` as grounding
   context, with instructions to answer only from them.

Embeddings are computed in-process here — unlike the
[API version](https://github.com/shashikirankulkarni/smart-query-backend),
which offloads them to a hosted endpoint to fit in a small container. Streamlit
Cloud gives enough memory to hold the model locally, which removes a network hop
and the timeout that comes with it.

Retrieval is what keeps this useful as the sheet grows: sending all rows to the
model would blow the context window and bury the relevant pair in noise.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml`:

```toml
COHERE_API_KEY = "your-cohere-api-key"
```

## Deploy on Streamlit Cloud

Push to GitHub, deploy via [streamlit.io/cloud](https://streamlit.io/cloud), and
add `COHERE_API_KEY` under **App settings → Secrets**. It is read with
`st.secrets` and never committed.

## Sheet format

One row per pair, with `Question` and `Answer` columns. `sample_qa.xlsx` in the
repo is a working example.

## Stack

Streamlit · Cohere `command-r+` · sentence-transformers · FAISS · pandas ·
openpyxl

## Limitations

- The FAISS index is rebuilt per session; a large sheet means a slow first load.
- Retrieval matches on the question column only — an answer containing the
  relevant term won't be found if its question doesn't.
- Chat history lives in `st.session_state` and resets when the session ends.
