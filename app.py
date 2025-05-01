import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import requests

st.set_page_config(page_title="Excel Q&A Chatbot", layout="centered")
st.title("📊 Excel Q&A Chatbot")

uploaded_file = st.file_uploader("Upload Excel (2 columns: 'Question', 'Answer')", type=["xlsx"])

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if not {'Question', 'Answer'}.issubset(df.columns):
        st.error("Excel must contain 'Question' and 'Answer' columns")
    else:
        questions = df['Question'].tolist()
        question_embeddings = model.encode(questions, convert_to_numpy=True)

        index = faiss.IndexFlatL2(question_embeddings.shape[1])
        index.add(np.array(question_embeddings))

        def search_context(query, top_k=3):
            query_embedding = model.encode([query], convert_to_numpy=True)
            D, I = index.search(query_embedding, top_k)
            return df.iloc[I[0]]

        def get_answer_with_ollama(query, context_df):
            context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(context_df['Question'], context_df['Answer'])])
            prompt = f"""Answer based ONLY on the following Q&A context. If not found, say \"I don't know.\"\n\nContext:\n{context}\n\nUser: {query}\nAnswer:"""
            try:
                res = requests.post("http://localhost:11434/api/generate", json={
                    "model": "mistral",
                    "prompt": prompt
                })
                return res.json().get('response', '[Error getting response]')
            except Exception as e:
                return f"Error connecting to LLM: {str(e)}"

        st.success("Excel loaded. Ask your question below.")
        query = st.text_input("Your Question")
        if query:
            context_df = search_context(query)
            answer = get_answer_with_ollama(query, context_df)
            st.markdown(f"**Answer:** {answer}")
