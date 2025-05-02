import streamlit as st
import pandas as pd
import cohere
import os
import requests
from io import BytesIO
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Smart Query Bot", layout="centered")

COHERE_API_KEY = st.secrets.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

st.title("💬 Smart Query Bot")

if "df_data" not in st.session_state:
    st.session_state.df_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

excel_url = st.text_input("Paste a public Excel/CSV/Google Sheet URL:")
if st.button("🔄 Sync File"):
    try:
        if "docs.google.com/spreadsheets" in excel_url:
            file_id = excel_url.split("/d/")[1].split("/")[0]
            csv_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
            df = pd.read_csv(BytesIO(requests.get(csv_url).content))
        elif excel_url.endswith(".csv"):
            df = pd.read_csv(BytesIO(requests.get(excel_url).content))
        else:
            if "drive.google.com" in excel_url and "uc?export=download" not in excel_url:
                file_id = excel_url.split("/d/")[1].split("/")[0]
                excel_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            elif "dropbox.com" in excel_url:
                excel_url = excel_url.replace("?dl=0", "?dl=1")
            response = requests.get(excel_url, timeout=15)
            response.raise_for_status()
            df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
        st.session_state.df_data = df
        st.session_state.chat_history = []
        st.success("✅ File loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to fetch or read file: {e}")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
df = st.session_state.df_data

if df is not None and {'Question', 'Answer'}.issubset(df.columns):

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

    st.markdown("---")
    st.subheader("🟢 Chat")

    # Display chat history
    for role, message in st.session_state.chat_history:
        align = "right" if role == "user" else "left"
        bg = "#dcf8c6" if role == "user" else "#f1f0f0"
        st.markdown(
            f"<div style='background-color:{bg}; padding:10px 15px; border-radius:10px; "
            f"margin:5px; max-width:80%; float:{align}; clear:both; text-align:left;'>"
            f"{message}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

    # Input field at the bottom
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message", "")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.chat_history.append(("user", user_input))
        context_df = search_context(user_input)
        response = call_cohere_chat(user_input, context_df)
        st.session_state.chat_history.append(("bot", response))
