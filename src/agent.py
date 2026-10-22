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

def retrieve_policy_context(query, top_k=4):
    if not chunk_texts or len(multilingual_embeddings) == 0:
        return []

    query_vector = embedding_model.encode(
        [str(query)],
        normalize_embeddings=True,
    )[0]

    scores = np.asarray(multilingual_embeddings) @ query_vector
    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in indices:
        score = float(scores[index])
        if score >= 0.20:
            results.append({
                "text": chunk_texts[int(index)],
                "source": chunk_sources[int(index)],
                "score": score,
            })

    return results

import re


def verify_customer(order_id, answer):
    answer = str(answer).strip()
    result = lookup_order(str(order_id))

    if not result.get("found"):
        return False

    order = result["order"]

    # Verify using the order's delivery pincode.
    if answer == str(order.get("delivery_pincode", "")).strip():
        return True

    # Or verify using the registered phone's last four digits.
    customer_id = str(order.get("customer_id", "")).strip()
    rows = customers[
        customers["customer_id"].astype(str) == customer_id
    ]

    if not rows.empty:
        phone = re.sub(r"\D", "", str(rows.iloc[0].get("phone", "")))
        supplied = re.sub(r"\D", "", answer)

        if len(phone) >= 4 and len(supplied) == 4:
            return supplied == phone[-4:]

    return False


def support_turn_v2(session_id, user_message):
    sid = str(session_id)
    msg = str(user_message).strip()
    low = msg.lower()

    state = conversation_state.setdefault(sid, {
        "history": [],
        "pending_verification": None,
        "verified_orders": set(),
        "pending_ticket": False,
    })

    if not msg:
        return {
            "reply": "Please tell me how I can help.",
            "retrieved_chunks": [],
        }

    # Handle a pending verification request.
    pending = state["pending_verification"]
    if pending:
        if verify_customer(pending, msg):
            state["verified_orders"].add(pending)
            state["pending_verification"] = None
            order = lookup_order(pending)["order"]
            reply = (
                f"Verification successful. Order {pending} status: "
                f"{order.get('status', 'unavailable')}. "
                f"Promised delivery date: "
                f"{order.get('promised_date', 'unavailable')}."
            )
        else:
            state["pending_verification"] = None
            reply = (
                "I couldn't verify those details. For your security, "
                "please contact customer support."
            )

        return {"reply": reply, "retrieved_chunks": []}

    # Handle ticket creation and follow-up descriptions.
    ticket_intent = any(term in low for term in (
        "create a ticket", "raise a ticket", "support ticket",
        "create ticket", "escalate this", "raise this issue",
    ))

    if state["pending_ticket"] or ticket_intent:
        summary = msg
        if ticket_intent:
            summary = re.sub(
                r"\b(create|raise|open)\s+(a\s+)?(support\s+)?ticket\b",
                "", msg, flags=re.I,
            ).strip(" .,:-")

        if len(summary) < 8:
            state["pending_ticket"] = True
            return {
                "reply": "Please briefly describe the issue for the support ticket.",
                "retrieved_chunks": [],
            }

        ticket_result = create_ticket(summary)
        state["pending_ticket"] = False

        if ticket_result.get("created") is True:
            reply = (
                "Your support ticket has been created. Reference: "
                f"{ticket_result['ticket_id']}."
            )
        else:
            reply = "I couldn't confirm ticket creation. Please contact support."

        return {"reply": reply, "retrieved_chunks": []}

    # Find an order ID in the message.
    order_id = None
    for oid in orders["order_id"].astype(str):
        if re.search(
            r"(?<![A-Za-z0-9])" + re.escape(oid) + r"(?![A-Za-z0-9])",
            msg, re.I,
        ):
            order_id = oid
            break

    if order_id:
        result = lookup_order(order_id)
        if not result.get("found"):
            reply = "I couldn't find that order ID. Please check it."
        elif order_id not in state["verified_orders"]:
            state["pending_verification"] = order_id
            reply = (
                f"Before I share details for order {order_id}, please verify "
                "using its delivery pincode or your registered phone's last four digits."
            )
        else:
            order = result["order"]
            reply = (
                f"Order {order_id} status: {order.get('status', 'unavailable')}. "
                f"Promised delivery date: {order.get('promised_date', 'unavailable')}."
            )

        state["history"].extend([
            {"role": "user", "content": msg},
            {"role": "assistant", "content": reply},
        ])
        return {"reply": reply, "retrieved_chunks": []}

    # Answer policy questions using retrieved evidence.
    retrieved = retrieve_policy_context(msg)
    context = "\n\n".join(
        f"[Source: {item['source']}]\n{item['text']}"
        for item in retrieved
    )

    system_prompt = (
        "You are a helpful, concise


        
