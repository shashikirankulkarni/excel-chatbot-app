# excel_chatbot_app/app.py

import streamlit as st
import pandas as pd
import cohere
import os
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Q&A Chatbot (Cohere)", layout="centered")
st.title("🤖 Chatbot from Uploaded File (Cohere)")

COHERE_API_KEY = st.secrets.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

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
        st.success("File loaded. Ask your question below.")

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
            ]
            prompt = f"""
You are a helpful assistant. Answer ONLY based on the following Q&A pairs.
If the answer is not available in this data, say: \"I don't know.\"

Q&A Context:
{chr(10).join([f"Q: {d['snippet'].split('\\n')[0][3:]}\nA: {d['snippet'].split('\\n')[1][3:]}" for d in documents])}

User: {query}
Answer:
"""
            try:
                response = co.generate(
                    model="command-r",
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.3
                )
                return response.generations[0].text.strip()
            except Exception as e:
                return f"[Cohere API Error] {e}"

        query = st.text_input("Ask your question")
        if query:
            context_df = search_context(query)
            answer = call_cohere_chat(query, context_df)
            st.markdown(f"**Answer:** {answer}")
