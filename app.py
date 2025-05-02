import streamlit as st
import pandas as pd
import cohere
import os
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Q&A Chatbot (Cohere)", layout="centered")
st.title("🤖 Chatbot from Shared Link (Cohere)")

COHERE_API_KEY = st.secrets.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# UI: Paste link to public Excel file
excel_url = st.text_input("Paste a public Excel file URL (Google Drive, Dropbox, etc.)")
sync_clicked = st.button("🔄 Sync")

df = None
if sync_clicked:
    if not excel_url:
        st.warning("Please paste a valid public link to an Excel file.")
    else:
        try:
            # Attempt to download and read the file
            if "docs.google.com/spreadsheets" in url:
                # Convert Google Sheet to CSV
                file_id = url.split("/d/")[1].split("/")[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
                df = pd.read_csv(BytesIO(requests.get(csv_url).content))
            elif url.endswith(".csv"):
                df = pd.read_csv(BytesIO(requests.get(url).content))
            else:
                df = pd.read_excel(BytesIO(requests.get(url).content))

            response = requests.get(excel_url, timeout=15)
            response.raise_for_status()
            df = pd.read_excel(BytesIO(response.content))
            st.success("✅ Excel file synced successfully!")

        except Exception as e:
            st.error(f"❌ Failed to fetch or read Excel file: {e}")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

if df is not None:
    if not {'Question', 'Answer'}.issubset(df.columns):
        st.error("Excel must contain 'Question' and 'Answer' columns")
    else:
        st.subheader("Ask Your Question")

        def search_context(query, top_k=3):
            query_embedding = model.encode([query], convert_to_tensor=True)
            corpus_embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)
            results = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
            top_indices = [hit['corpus_id'] for hit in results[0]]
            return df.iloc[top_indices]

        def call_cohere_chat(query, context_df):
            documents = [
                {"title": f"Q{i+1}", "snippet": f"Q: {q}\nA: {a}"}
                for i, (q, a) in enumerate(zip(context_df['Question'], context_df['Answer']))
                if pd.notna(q) and pd.notna(a)
            ]

            preamble = (
                "You are a helpful assistant. Answer ONLY based on the following Q&A pairs. "
                "If the answer is not available in this data, say: 'I don't know.'"
            )

            try:
                response = co.chat(
                    model="command-r",
                    message=query,
                    documents=documents,
                    preamble=preamble,
                    temperature=0.3
                )
                return response.text.strip()
            except Exception as e:
                return f"[Cohere API Error] {e}"

        query = st.text_input("Enter your question")
        if query:
            context_df = search_context(query)
            answer = call_cohere_chat(query, context_df)
            st.markdown(f"**Answer:** {answer}")
