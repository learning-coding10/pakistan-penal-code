# streamlit_app.py
import os
import streamlit as st
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import json
from tqdm import tqdm
from dotenv import load_dotenv
import textwrap

load_dotenv()

# ---------------------------
# Config
# ---------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for inference API
MODEL_NAME = "openai/gpt-oss-20b"  # recommended open model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PDF_FOLDER = "PAKISTAN PENAL CODE.pdf"
CHUNK_SIZE = 1000   # characters per chunk (you can tune)
CHUNK_OVERLAP = 200
TOP_K = 5
LOCAL_MODEL = False  # set True if you will run a local model with transformers (needs GPU & setup)

# ---------------------------
# Helpers: PDF -> text
# ---------------------------
def pdf_to_text(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for p in doc:
        pages.append(p.get_text("text"))
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end == L:
            break
        start = end - overlap
    return chunks

# ---------------------------
# Indexing: Build embeddings + FAISS
# ---------------------------
@st.cache_data(show_spinner=False)
def build_index(pdf_folder: str):
    files = []
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith(".pdf"):
            files.append(os.path.join(pdf_folder, fname))
    if not files:
        return None

    doc_texts = []
    doc_sources = []  # (filename, chunk_id)
    for f in files:
        text = pdf_to_text(f)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            doc_texts.append(c)
            doc_sources.append((os.path.basename(f), i+1))  # 1-index chunk id

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embed_model.encode(doc_texts, show_progress_bar=True, convert_to_numpy=True)
    d = embeddings.shape[1]

    # FAISS index
    index = faiss.IndexFlatIP(d)
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # store metadata
    return {
        "index": index,
        "texts": doc_texts,
        "sources": doc_sources,
        "embed_model": embed_model
    }

# ---------------------------
# Retrieval
# ---------------------------
def retrieve(query: str, idx_struct, top_k: int = TOP_K) -> List[Tuple[str, Tuple[str,int], float]]:
    embed_model = idx_struct["embed_model"]
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = idx_struct["index"].search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        text = idx_struct["texts"][idx]
        source = idx_struct["sources"][idx]
        results.append((text, source, float(score)))
    return results

# ---------------------------
# Generation: call HF Inference API (default)
# ---------------------------
def generate_with_hf(prompt: str, max_new_tokens: int = 512, temperature: float = 0.1):
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set. Export your Hugging Face token as HF_TOKEN")
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
            # you can customize other parameters
        }
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"HF inference error {response.status_code}: {response.text}")
    resp = response.json()
    # HF inference returns list of generated sequences (depends on model)
    if isinstance(resp, dict) and "error" in resp:
        raise RuntimeError(f"HF inference error: {resp['error']}")
    # typical: [{'generated_text': '...'}]
    text = ""
    if isinstance(resp, list) and len(resp) > 0:
        text = resp[0].get("generated_text", "")
    elif isinstance(resp, dict):
        text = resp.get("generated_text", "")
    return text

# ---------------------------
# Prompt template (instruct to act like PPC expert)
# ---------------------------
PROMPT_SYSTEM = """
You are a legal assistant with deep expertise in the Pakistan Penal Code (PPC). 
Answer the user's question primarily by using ONLY the provided retrieved document chunks. 
Cite the source chunk numbers in square brackets after each citation (format: [filename.pdf, chunk #]). 
If the retrieved chunks are insufficient, clearly say "Insufficient info in provided documents — consult a legal expert" and avoid inventing laws. 
Provide concise, structured answers with section references where possible and a short actionable summary.
"""

def build_prompt(user_question: str, retrieved: List[Tuple[str, Tuple[str,int], float]]):
    # Build context section with short numbered chunks
    context_lines = []
    for i, (text, source, score) in enumerate(retrieved, start=1):
        filename, chunk_id = source
        header = f"[{i}] Source: {filename}, chunk {chunk_id} (score={score:.3f})"
        snippet = textwrap.shorten(text.replace("\n", " "), width=800, placeholder=" ...")
        context_lines.append(f"{header}\n{snippet}\n")
    context = "\n\n".join(context_lines)
    instructions = PROMPT_SYSTEM + "\n\n" + "Retrieved chunks:\n" + context + "\n\n"
    instructions += "User question: " + user_question + "\n\n"
    # Ask model to produce answer and include citations corresponding to chunk numbers
    instructions += "Answer now. Use the chunk numbering (e.g., [1]) to indicate which retrieved chunk you used. If you need to quote the chunk, put the quote in quotes and then cite the chunk."
    return instructions

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="PPC RAG Chatbot", layout="wide")
st.title("Pakistan Penal Code — RAG Chatbot (PDF-based)")

st.sidebar.header("Setup / Info")
st.sidebar.markdown("""
- Drop Pakistan Penal Code PDFs into the `pdfs/` directory.
- This app uses local embeddings (sentence-transformers) + FAISS.
- Default model for generation: **openai/gpt-oss-20b** via Hugging Face Inference API.
- Set environment variable `HF_TOKEN` (Hugging Face token) to invoke the model.
""")
st.sidebar.markdown("**Legal disclaimer:** This tool is informational only and not legal advice.")

with st.expander("Index / Build status", expanded=True):
    st.write("Building or loading index from `./pdfs/` ...")
    try:
        idx_struct = build_index(PDF_FOLDER)
        if idx_struct is None:
            st.warning("No PDFs found in ./pdfs/. Please upload Pakistan Penal Code PDFs into that folder and refresh.")
        else:
            st.success(f"Index built. {len(idx_struct['texts'])} chunks indexed from PDFs in `{PDF_FOLDER}`.")
    except Exception as e:
        st.error(f"Failed to build index: {e}")
        idx_struct = None

query = st.text_area("Enter your legal question about the Pakistan Penal Code (PPC):", height=150)

col1, col2 = st.columns([1,3])
with col1:
    top_k = st.number_input("Top K retrieval", min_value=1, max_value=10, value=TOP_K)
    temp = st.slider("Generation temperature", min_value=0.0, max_value=1.0, value=0.1)
    max_tokens = st.number_input("Max generated tokens", min_value=64, max_value=2048, value=512)
    run_btn = st.button("Get Answer")
with col2:
    st.write("Retrieved context and AI answer will appear here...")

if run_btn:
    if not query.strip():
        st.warning("Please enter a question.")
    elif idx_struct is None:
        st.error("No index exists. Place PDFs in ./pdfs/ and restart app.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve(query, idx_struct, top_k=top_k)
        if not retrieved:
            st.error("No relevant chunks found.")
        else:
            st.subheader("Top retrieved chunks")
            for i, (text, src, score) in enumerate(retrieved, start=1):
                fname, chunk_id = src
                st.markdown(f"**[{i}] {fname} — chunk {chunk_id}** (score {score:.3f})")
                st.write(textwrap.shorten(text.replace("\n", " "), width=600, placeholder=" ..."))
                st.markdown("---")

            # Build prompt
            prompt = build_prompt(query, retrieved)
            st.subheader("Constructed prompt (for debugging)")
            with st.expander("Show full prompt"):
                st.code(prompt[:4000] + ("\n\n...TRUNCATED..." if len(prompt) > 4000 else ""), language="text")

            st.subheader("Model answer")
            try:
                if LOCAL_MODEL:
                    # Placeholder: local model generation code would go here (requires substantial GPU)
                    answer = "LOCAL MODEL generation not configured in this sample. Set LOCAL_MODEL=False or implement local generation."
                else:
                    answer = generate_with_hf(prompt, max_new_tokens=max_tokens, temperature=temp)
                st.write(answer)
            except Exception as e:
                st.error(f"Generation failed: {e}")
