#!/usr/bin/env python
"""
rag_demo_splitter_history.py
────────────────────────────
Minimal Retrieval-Augmented-Generation demo with:

  • RecursiveCharacterTextSplitter  (LangChain utility only)
  • PDF + TXT ingestion
  • FAISS vector search
  • OpenAI embeddings + chat
  • Chat history inside the loop

Dependencies
------------
pip install openai==1.* faiss-cpu numpy pypdf langchain tqdm
"""

from __future__ import annotations
import os, json, textwrap
from pathlib import Path

import faiss, numpy as np, openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader               # <─ PDF extractor
from tqdm.auto import tqdm                # optional progress bar
from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────── Config ──────────────────────────────
DOCS_DIR        = Path(r"C:\Users\STEVE\Documents\GenAI\RAG_DATA")             # put .txt & .pdf files here
EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
CHUNK_SIZE      = 800                     # characters
CHUNK_OVERLAP   = 150
TOP_K           = 4

SYSTEM_PROMPT = (
    "You are a precise, concise tutor. "
    "Answer ONLY from the provided context. "
    "If the answer is missing, say “I don't know.”"
)

openai.api_key = os.getenv("openai_key")
assert openai.api_key, "👉  Please set OPENAI_API_KEY first!"
# ─────────────────────────────────────────────────────────────────

# 1️⃣  Load & split ----------------------------------------------------------
def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_and_split() -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks: list[str] = []
    for path in DOCS_DIR.rglob("*"):
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            text = read_pdf_text(path)
        else:
            continue
        chunks.extend(splitter.split_text(text))
    if not chunks:
        raise RuntimeError(f"No .txt or .pdf files found inside {DOCS_DIR}")
    return chunks

# 2️⃣  OpenAI embeddings ------------------------------------------------------
def embed(texts: list[str]) -> list[list[float]]:
    res = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

# 3️⃣  Build (or load) FAISS --------------------------------------------------
def get_faiss_store(chunks: list[str], idx_path: str = "faiss.index"):
    meta_path = idx_path + ".meta.json"
    if Path(idx_path).exists() and Path(meta_path).exists():
        print("✓  Loading existing vector store …")
        index = faiss.read_index(idx_path)
        chunks = json.loads(Path(meta_path).read_text())
        return index, chunks

    print("⏳  Building vector store …")
    all_vectors = []
    for i in tqdm(range(0, len(chunks), 128), unit="batch"):
        all_vectors.extend(embed(chunks[i : i + 128]))
    mat = np.asarray(all_vectors, dtype=np.float32)

    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, idx_path)
    Path(meta_path).write_text(json.dumps(chunks))
    return index, chunks

# 4️⃣  Retrieval --------------------------------------------------------------
def retrieve(query: str, index, chunks, k: int = TOP_K) -> list[str]:
    q_vec = np.asarray(embed([query])[0], dtype=np.float32).reshape(1, -1)
    _, idxs = index.search(q_vec, k)
    return [chunks[i] for i in idxs[0]]

# 5️⃣  Prompt helpers ---------------------------------------------------------
def build_user_prompt(question: str, ctx_chunks: list[str]) -> str:
    context_block = "\n\n".join(
        f"[Doc {i+1}]\n{chunk}" for i, chunk in enumerate(ctx_chunks)
    )
    return (
        "Use the context below to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )

# 6️⃣  Chat loop with history -------------------------------------------------
def chat_loop(index, chunks):
    history: list[dict] = []                 # stores past turns (user & assistant)
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}

    while True:
        try:
            q = input("\n💬  Ask (Ctrl-C to quit): ")
        except KeyboardInterrupt:
            print("\nBye!")
            break

        ctx = retrieve(q, index, chunks)
        user_prompt = build_user_prompt(q, ctx)

        # Combine system prompt, history, and current user input
        messages = [system_msg] + history + [{"role": "user", "content": user_prompt}]

        # Show the retrieved context (for teaching transparency)
        print("\n🔍  Retrieved context:")
        print("─" * 60)
        for i, c in enumerate(ctx, 1):
            print(textwrap.indent(textwrap.fill(c, width=88), f"[Doc {i}] "))
        print("─" * 60)

        # Call the chat model
        response = openai.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.2
        )
        answer = response.choices[0].message.content

        print("🤖  Answer:\n")
        print(textwrap.fill(answer, width=88))

        # Update history with the *plain* question and answer (omit long context)
        history.extend([
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer},
        ])

# ──────────────────────────── main ───────────────────────────────
if __name__ == "__main__":
    chunks = load_and_split()
    index, chunks = get_faiss_store(chunks)
    chat_loop(index, chunks)
