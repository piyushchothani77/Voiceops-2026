import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

DATA_ROOT = Path(
    os.getenv(
        "VOICEOPS_DATA_DIR",
        "Capstone_Project_Dataset_FTTT_2026",
    )
).expanduser().resolve()

if not (DATA_ROOT / "data").is_dir():
    raise FileNotFoundError(
        "Dataset not found. Set VOICEOPS_DATA_DIR to the extracted "
        "Capstone_Project_Dataset_FTTT_2026 directory."
    )

if not (DATA_ROOT / "kb").is_dir():
    raise FileNotFoundError(
        f"Knowledge-base folder not found: {DATA_ROOT / 'kb'}"
    )

# Import the official dataset tools.
sys.path.insert(0, str(DATA_ROOT))
from tools import lookup_order, create_ticket

# ============================================================
# 2. LOAD CURRENT POLICY DOCUMENTS
# ============================================================

documents = []

for file in (DATA_ROOT / "kb").rglob("*.md"):
    if "superseded" in file.name.lower():
        continue

    documents.append({
        "source": file.name,
        "text": file.read_text(
            encoding="utf-8",
            errors="ignore",
        ),
    })

chunk_texts = []
chunk_sources = []

for document in documents:
    text = document["text"]

    # Overlapping chunks preserve some surrounding context.
    for start in range(0, len(text), 850):
        chunk = text[start:start + 1000].strip()

        if chunk:
            chunk_texts.append(chunk)
            chunk_sources.append(document["source"])

if not chunk_texts:
    raise RuntimeError(
        "No policy chunks found. Check the dataset kb/ folder."
    )

# ============================================================
# 3. INITIALIZE EMBEDDINGS AND GROQ
# ============================================================

print("Loading multilingual embedding model...")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

multilingual_embeddings = embedding_model.encode(
    chunk_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Configure it in your local .env file."
    )

client = Groq(api_key=api_key)

# Separate conversation state for each session.
conversation_state = {}

print(f"Loaded {len(chunk_texts)} policy chunks.")
print("VoiceOps agent initialized.")


# ============================================================
# 4. POLICY RETRIEVAL
# ============================================================

def retrieve_policy_context(query, top_k=4):
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


# ============================================================
# 5. CONVERSATION HANDLER
# ============================================================

def support_turn_v2(session_id, user_message):
    sid = str(session_id)
    msg = str(user_message).strip()
    low = msg.lower()

    state = conversation_state.setdefault(
        sid,
        {
            "history": [],
            "pending_verification": None,
            "verified_orders": set(),
            "pending_ticket": None,
        },
    )

    if not msg:
        return {
            "reply": "Please tell me how I can help.",
            "retrieved_chunks": [],
        }

    # --------------------------------------------------------
    # A. Handle an ongoing order-verification conversation.
    # --------------------------------------------------------

    pending = state["pending_verification"]

    if pending is not None:
        order_id = pending["order_id"]

        # The official tool validates the supplied proof.
        result = lookup_order(order_id, verification_value=msg)

        if result.get("found") is True and result.get("verified") is True:
            state["verified_orders"].add(order_id)
            state["pending_verification"] = None

            reply = (
                f"Verification successful. "
                f"Order {order_id} status: "
                f"{result.get('status') or 'unavailable'}. "
                f"Promised delivery date: "
                f"{result.get('promised_date') or 'unavailable'}."
            )

        else:
            pending["attempts"] += 1

            if pending["attempts"] <= 1:
                reply = (
                    "Those details did not match. You have one more attempt. "
                    "Please provide the registered phone's last four digits "
                    "or the delivery pincode."
                )
            else:
                state["pending_verification"] = None
                reply = (
                    "I couldn't verify those details. For your security, "
                    "please contact customer support."
                )

        state["history"].extend([
            {"role": "user", "content": msg},
            {"role": "assistant", "content": reply},
        ])

        return {
            "reply": reply,
            "retrieved_chunks": [],
        }

    # --------------------------------------------------------
    # B. Handle ticket creation and follow-up descriptions.
    # --------------------------------------------------------

    ticket_intent = any(
        phrase in low
        for phrase in (
            "create a ticket",
            "raise a ticket",
            "support ticket",
            "create ticket",
            "escalate this",
            "raise this issue",
            "talk to a human",
            "speak to a human",
            "human agent",
        )
    )

    pending_ticket = state["pending_ticket"]

    if pending_ticket is not None or ticket_intent:
        if pending_ticket is not None:
            summary = (
                pending_ticket["summary"] + " " + msg
            ).strip()
            ticket_details = pending_ticket
        else:
            summary = re.sub(
                r"\b(create|raise|open)\s+(a\s+)?"
                r"(support\s+)?ticket\b",
                "",
                msg,
                flags=re.IGNORECASE,
            ).strip(" .,:-")

            ticket_details = {
                "order_id": None,
                "intent": "support_escalation",
                "reason": "Customer requested human assistance",
            }

        if len(summary) < 20:
            state["pending_ticket"] = {
                **ticket_details,
                "summary": summary,
            }

            return {
                "reply": (
                    "Please describe the issue in a little more detail "
                    "so I can prepare the support handover."
                ),
                "retrieved_chunks": [],
            }

        ticket_result = create_ticket(
            summary=summary,
            order_id=ticket_details.get("order_id"),
            reason=ticket_details.get("reason"),
            intent=ticket_details.get("intent"),
        )

        state["pending_ticket"] = None

        if ticket_result.get("created") is True:
            reply = (
                "Your support ticket has been created. "
                f"Reference: {ticket_result['ticket_id']}."
            )
        else:
            reply = (
                "I couldn't confirm ticket creation. "
                "Please contact customer support."
            )

        state["history"].extend([
            {"role": "user", "content": msg},
            {"role": "assistant", "content": reply},
        ])

        return {
            "reply": reply,
            "retrieved_chunks": [],
        }

    # --------------------------------------------------------
    # C. Detect order IDs without revealing order information.
    # --------------------------------------------------------

    # The dataset's order IDs are numeric. Do not query the tool
    # for order details until the customer supplies verification.
    order_match = re.search(r"(?<!\d)\d{6,}(?!\d)", msg)

    if order_match:
        order_id = order_match.group(0)

        if order_id not in state["verified_orders"]:
            state["pending_verification"] = {
                "order_id": order_id,
                "attempts": 0,
            }

            reply = (
                "Before I share order details, please verify your identity "
                "using the registered phone's last four digits "
                "or the delivery pincode."
            )

            state["history"].extend([
                {"role": "user", "content": msg},
                {"role": "assistant", "content": reply},
            ])

            return {
                "reply": reply,
                "retrieved_chunks": [],
            }

    # --------------------------------------------------------
    # D. Retrieve policy evidence and answer using Groq.
    # --------------------------------------------------------

    retrieved = retrieve_policy_context(msg)

    context = "\n\n".join(
        f"[Source: {item['source']} | "
        f"similarity: {item['score']:.3f}]\n"
        f"{item['text']}"
        for item in retrieved
    )

    system_prompt = (
        "You are VoiceOps, a helpful and concise Indian e-commerce "
        "support agent for the fictional company ZentroMart. "
        "Use only the retrieved policy excerpts for policy-specific claims. "
        "If evidence is missing or ambiguous, say you cannot confirm "
        "and offer to escalate. Never invent eligibility, refund timelines, "
        "dates, order statuses, or completed actions. Never reveal order "
        "details without successful verification. Never claim a ticket "
        "was created unless the official tool returned created=true. "
        "Keep replies short and natural for spoken delivery. "
        "Respond in the customer's language when possible.\n\n"
        "RETRIEVED POLICY EXCERPTS:\n"
        + (
            context
            if context
            else "[No sufficiently relevant policy chunks retrieved.]"
        )
    )

    messages = (
        [{"role": "system", "content": system_prompt}]
        + state["history"][-8:]
        + [{"role": "user", "content": msg}]
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2,
    )

    reply = response.choices[0].message.content.strip()

    state["history"].extend([
        {"role": "user", "content": msg},
        {"role": "assistant", "content": reply},
    ])

    return {
        "reply": reply,
        "retrieved_chunks": [
            {
                "source": item["source"],
                "score": round(item["score"], 3),
                "text": item["text"],
            }
            for item in retrieved
        ],
    }
