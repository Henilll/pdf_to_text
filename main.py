import streamlit as st
import pdfplumber
from groq import Groq
import numpy as np

# --- UI Settings ---
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="centered"
)

st.markdown("""
# 📄 PDF Q&A Chatbot
Ask questions directly from your PDF documents!  
""")

# --- Static API Key ---
GROQ_API_KEY = "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"  # <-- Replace with your Groq API key
client = Groq(api_key=GROQ_API_KEY)

# --- Request Counter ---
if "requests_made" not in st.session_state:
    st.session_state.requests_made = 0
MAX_REQUESTS = 100

# --- Upload PDF ---
uploaded = st.file_uploader("Upload your PDF here", type="pdf", label_visibility="visible")

# --- Extract Text ---
def extract_text(pdf_bytes):
    text = ""
    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# --- Simple embedding for lightweight vector search ---
def simple_embed(text, vocab=None):
    words = text.lower().split()
    if vocab is None:
        vocab = list(set(words))
    vec = np.array([words.count(w) for w in vocab], dtype=np.float32)
    if np.linalg.norm(vec) > 0:
        vec = vec / np.linalg.norm(vec)
    return vec, vocab

def search(query_vec, doc_vecs, top_k=5):
    sims = [np.dot(query_vec, dv) for dv in doc_vecs]
    idx = np.argsort(sims)[-top_k:][::-1]
    return idx

# --- Main App ---
if uploaded:
    with st.spinner("📖 Extracting text from PDF..."):
        text = extract_text(uploaded)

    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    doc_vecs = []
    vocab = None
    for c in chunks:
        v, vocab = simple_embed(c, vocab)
        doc_vecs.append(v)
    doc_vecs = np.array(doc_vecs)

    st.success("✅ PDF processed successfully!")

    st.markdown("### Ask a question about your PDF:")
    question = st.text_input("Type your question here", placeholder="e.g., What are the main points of section 2?")

    if st.button("Get Answer"):
        if st.session_state.requests_made >= MAX_REQUESTS:
            st.error("⚠️ Daily limit of 100 requests reached!")
        elif question.strip() == "":
            st.error("Please enter a question.")
        else:
            st.session_state.requests_made += 1
            q_vec, _ = simple_embed(question, vocab)
            top_idx = search(q_vec, doc_vecs)
            context = "\n\n---\n\n".join([chunks[i] for i in top_idx])

            with st.spinner("🤖 Generating answer from Groq..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Answer ONLY using the given PDF context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                    ]
                )

            st.markdown("### 📝 Answer:")
            st.text_area("", value=response.choices[0].message.content, height=250)

            st.info(f"Requests used: {st.session_state.requests_made}/{MAX_REQUESTS}")

# --- Footer ---
st.markdown("---")
st.markdown("<center>Made with ❤️ by Henil</center>", unsafe_allow_html=True)
