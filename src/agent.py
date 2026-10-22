import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# Dataset location: set VOICEOPS_DATA_DIR to the extracted dataset folder.
DATA_ROOT = Path(
    os.getenv("VOICEOPS_DATA_DIR", "Capstone_Project_Dataset_FTTT_2026")
)

if not (DATA_ROOT / "data").is_dir():
    raise FileNotFoundError(
        "Dataset not found. Set VOICEOPS_DATA_DIR to the extracted "
        "Capstone_Project_Dataset_FTTT_2026 folder."
    )

# Load tables with IDs preserved as strings.
tables = {
    f.stem: pd.read_csv(f, dtype=str).fillna("")
    for f in (DATA_ROOT / "data").glob("*.csv")
}

orders = tables["orders"]
customers = tables["customers"]
order_items = tables["order_items"]

# Load current policy documents; skip superseded files.
documents = []
for file in (DATA_ROOT / "kb").rglob("*.md"):
    if "superseded" in file.name.lower():
        continue
    documents.append({
        "source": file.name,
        "text": file.read_text(encoding="utf-8", errors="ignore"),
    })

chunk_texts = []
chunk_sources = []

for document in documents:
    text = document["text"]
    for start in range(0, len(text), 850):
        chunk = text[start:start + 1000].strip()
        if chunk:
            chunk_texts.append(chunk)
            chunk_sources.append(document["source"])

if not chunk_texts:
    raise RuntimeError("No policy Markdown files found in the dataset.")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)
multilingual_embeddings = embedding_model.encode(
    chunk_texts, normalize_embeddings=True
)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set GROQ_API_KEY in your local environment.")

client = Groq(api_key=api_key)

# Initialize the mock tools and agent state.
conversation_state = {}
tickets = []


def lookup_order(order_id):
    match = orders[orders["order_id"] == str(order_id)]
    if match.empty:
        return {"found": False}

    order = match.iloc[0].to_dict()
    items = order_items[
        order_items["order_id"] == str(order_id)
    ].to_dict("records")

    return {"found": True, "order": order, "items": items}


def create_ticket(summary):
    import uuid
    ticket_id = "ZT-" + uuid.uuid4().hex[:8].upper()
    tickets.append({"ticket_id": ticket_id, "summary": summary})
    return {"created": True, "ticket_id": ticket_id}


print("VoiceOps dataset, embeddings, and Groq client initialized.")
