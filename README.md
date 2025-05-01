# Excel Q&A Chatbot (Cohere + Streamlit Cloud)

This is a chatbot that:
- Accepts an Excel file with Q&A pairs
- Uses Cohere's `command-r+` model for intelligent responses
- Hosted publicly using [Streamlit Cloud](https://streamlit.io/cloud)

## Setup Instructions

### 1. Add Secrets in Streamlit Cloud

In your app settings → `Secrets`:

```
COHERE_API_KEY = "your-cohere-api-key"
```

### 2. Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy on Streamlit Cloud

- Push this repo to GitHub
- Deploy it via [streamlit.io/cloud](https://streamlit.io/cloud)
