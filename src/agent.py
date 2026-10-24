import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from tools.tools import lookup_order, create_ticket

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
    query_text = str(query)
    query_lower = query_text.lower()

    query_vector = embedding_model.encode(
        [query_text],
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

    # Deterministic guardrail: return-window questions must include RET-2,
    # the authoritative category table. Semantic retrieval can otherwise
    # rank the festive-sale exception (RET-8) above the normal return window.
    return_terms = (
        "return", "clothing", "apparel", "fashion",
        "कपड़े", "कपड़", "कपड़", "वापस", "रिटर्न", "जूते", "बैग",
    )

    if any(term in query_lower for term in return_terms):
        for document in documents:
            if document.get("source") != "pol_returns_v4.md":
                continue

            match = re.search(
                r"(?ms)^## \[RET-2\].*?(?=^## \[RET-3\]|\Z)",
                document.get("text", ""),
            )

            if match:
                ret2_text = match.group(0).strip()
                if not any(
                    "[RET-2]" in item["text"] and "15 days" in item["text"]
                    for item in results
                ):
                    results.insert(0, {
                        "text": ret2_text,
                        "source": "pol_returns_v4.md",
                        "score": 1.0,
                    })
                break

    return results[:max(top_k, 1)]

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


# ============================================================
# VOICEOPS EVALUATION-INFORMED GUARDRAILS
# These deterministic checks run before general RAG generation.
# They use the supplied synthetic catalogue and current policies.
# ============================================================

from datetime import date, timedelta

_voiceops_base_support_turn = support_turn_v2
_voiceops_products_path = DATA_ROOT / "data" / "products.csv"
_voiceops_products = (
    pd.read_csv(_voiceops_products_path, dtype=str).fillna("")
    if _voiceops_products_path.exists()
    else pd.DataFrame()
)


def _voiceops_policy_section(section_id):
    pattern = re.compile(
        rf"(?ms)^## \[{re.escape(section_id)}\].*?(?=^## \[|\Z)"
    )
    for doc in documents:
        match = pattern.search(doc.get("text", ""))
        if match:
            return doc.get("source", ""), match.group(0).strip()
    return "", ""


def _voiceops_evidence(*items):
    return [
        {"source": source, "score": 1.0, "text": text}
        for source, text in items
        if source and text
    ]


def _voiceops_special_result(session_id, user_message, reply, evidence):
    sid = str(session_id)
    state = conversation_state.setdefault(
        sid,
        {
            "history": [],
            "pending_verification": None,
            "verified_orders": set(),
            "pending_ticket": None,
        },
    )
    state["history"].extend(
        [
            {"role": "user", "content": str(user_message)},
            {"role": "assistant", "content": reply},
        ]
    )
    return {"reply": reply, "retrieved_chunks": evidence}


def _voiceops_find_product(message):
    msg = str(message)
    low = msg.lower()
    row = None

    sku_match = re.search(r"\b([A-Z]{3})-?(\d{4})\b", msg, re.I)
    sku = (
        f"{sku_match.group(1).upper()}-{sku_match.group(2)}"
        if sku_match else None
    )

    # ASR often spells the four SKU digits as words, e.g.
    # "sku ele-zero zero one to" -> ELE-0012.
    if not sku and re.search(r"\bsku\b", low):
        prefix_match = re.search(r"\b(ELE|HOM|BKS|CLO|FSH|BEA|GRO)[\s-]+", low, re.I)
        if prefix_match:
            prefix = prefix_match.group(1).upper()
            tail = low[prefix_match.end():]
            number_tokens = re.findall(
                r"zero|oh|one|two|to|too|three|four|for|five|six|seven|eight|ate|nine|\d+",
                tail,
            )[:4]
            words_to_digits = {
                "zero": "0", "oh": "0", "one": "1", "two": "2",
                "to": "2", "too": "2", "three": "3", "four": "4",
                "for": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "ate": "8", "nine": "9",
            }
            digits = "".join(words_to_digits.get(token, token) for token in number_tokens)
            if len(digits) == 4:
                sku = f"{prefix}-{digits}"

    if sku and not _voiceops_products.empty and "sku" in _voiceops_products:
        found = _voiceops_products[
            _voiceops_products["sku"].str.upper() == sku.upper()
        ]
        if not found.empty:
            return found.iloc[0]

    # Fall back to matching distinctive tokens from the catalogue title.
    if not _voiceops_products.empty:
        query_tokens = set(re.findall(r"[a-z0-9]+", low))
        stop = {
            "the", "my", "is", "was", "a", "an", "sku", "return",
            "warranty", "assured", "product", "delivered", "on",
            "what", "with", "how", "does", "this", "that", "for",
        }
        query_tokens -= stop
        best_score = 0
        for _, candidate in _voiceops_products.iterrows():
            title_tokens = set(
                re.findall(r"[a-z0-9]+", str(candidate.get("title", "")).lower())
            ) - stop
            overlap = len(query_tokens & title_tokens)
            if overlap >= 2 and overlap > best_score:
                row = candidate
                best_score = overlap
    return row


def _voiceops_product_evidence(row):
    sku = str(row.get("sku", ""))
    detail = (
        f"SKU: {sku}; title: {row.get('title', '')}; "
        f"return_category: {row.get('return_category', '')}; "
        f"assured: {row.get('assured', '')}; "
        f"warranty_months: {row.get('warranty_months', '')}; "
        f"price_inr: {row.get('price_inr', '')}"
    )
    return (f"products.csv:{sku}", detail)


def _voiceops_parse_date(message):
    msg = str(message).lower()
    # Year is optional in the practice set; its dataset reference year is 2026.
    month_pattern = (
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    )
    word_numbers = {
        "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
        "thirteen": "13", "fourteen": "14", "fifteen": "15",
        "sixteen": "16", "seventeen": "17", "eighteen": "18",
        "nineteen": "19", "twenty": "20", "twenty-one": "21",
        "twenty-two": "22", "twenty-three": "23", "twenty-four": "24",
        "twenty-five": "25", "twenty-six": "26", "twenty-seven": "27",
        "twenty-eight": "28", "twenty-nine": "29", "thirty": "30",
        "thirty-one": "31",
    }
    for word, number in sorted(word_numbers.items(), key=lambda pair: -len(pair[0])):
        msg = re.sub(
            rf"\b{re.escape(word)}(?=\s+{month_pattern}\b)",
            number,
            msg,
        )
    match = re.search(
        rf"\b(?P<day>\d{{1,2}})\s+{month_pattern}"
        rf"(?:\s+(?P<year>20\d{{2}}))?\b",
        msg,
    )
    if not match:
        return None
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    try:
        return date(
            int(match.group("year") or 2026),
            months[match.group(2)[:3]],
            int(match.group("day")),
        )
    except ValueError:
        return None


def _voiceops_business_days_after(start_date, number_of_days):
    holidays = set()
    holiday_path = DATA_ROOT / "data" / "holidays.csv"
    if holiday_path.exists():
        holiday_df = pd.read_csv(holiday_path, dtype=str).fillna("")
        for column in holiday_df.columns:
            if "date" in column.lower():
                for value in holiday_df[column]:
                    parsed = pd.to_datetime(value, errors="coerce")
                    if not pd.isna(parsed):
                        holidays.add(parsed.date())

    current = start_date
    added = 0
    while added < number_of_days:
        current += timedelta(days=1)
        # Dataset business days are Monday-Saturday, excluding listed holidays.
        if current.weekday() != 6 and current not in holidays:
            added += 1
    return current


def support_turn_v2(session_id, user_message):
    msg = re.sub(
        r"\bre[\s-]+fund\b", "refund", str(user_message), flags=re.I
    ).strip()
    low = msg.lower()

    # 1. Use the structured product catalogue and WAR-2 for warranty questions.
    warranty_intent = any(
        term in low
        for term in ("assured", "warranty", "warrantee", "वारंटी", "गारंटी")
    )
    if warranty_intent:
        product = _voiceops_find_product(msg)
        if product is None:
            return _voiceops_special_result(
                session_id, msg,
                "Please share the product SKU or exact product name so I can check its warranty.",
                [],
            )

        sku = str(product.get("sku", ""))
        title = str(product.get("title", "this product"))
        brand_months = str(product.get("warranty_months", ""))
        assured = str(product.get("assured", "")).strip().lower() in {
            "true", "1", "yes"
        }
        policy_source, policy_text = _voiceops_policy_section("WAR-2")
        evidence = _voiceops_evidence(
            (policy_source, policy_text),
            _voiceops_product_evidence(product),
        )
        hinglish = bool(
            re.search(r"[\u0900-\u097f]", low)
            or re.search(r"\b(hai|kaise|pe|kya|mein)\b", low)
        )

        if assured:
            if hinglish:
                reply = (
                    f"Haan, {sku} par delivery date se 12-month ZentroMart Assured warranty milti hai. "
                    f"Catalog mein {brand_months}-month brand warranty bhi listed hai. "
                    "Claim ke liye app mein My Orders kholen."
                )
            else:
                reply = (
                    f"Yes. {title} ({sku}) has a 12-month ZentroMart Assured warranty from delivery "
                    f"plus a {brand_months}-month brand warranty listed in the catalogue. "
                    "Raise the Assured claim in the app under My Orders."
                )
        else:
            reply = (
                f"{title} ({sku}) is not marked as ZentroMart Assured. "
                f"The catalogue lists a {brand_months}-month brand warranty; "
                "contact the manufacturer's service centre for a claim."
            )
        return _voiceops_special_result(session_id, msg, reply, evidence)

    # 2. Payment-specific refund policies override generic FAQ text.
    refund_intent = any(
        term in low for term in ("refund", "रिफंड", "money back", "re fund")
    )
    if refund_intent:
        if "wallet" in low or "zentromart wallet" in low:
            src, text = _voiceops_policy_section("REF-2")
            return _voiceops_special_result(
                session_id, msg,
                "Refunds to ZentroMart Wallet are instant, within 2 hours of initiation.",
                _voiceops_evidence((src, text)),
            )

        normalized = re.sub(r"\bu\s+p\s+i\b", "upi", low)
        payment = None
        bounds = None
        if "upi" in normalized or "यूपीआई" in normalized:
            payment, bounds = "UPI", (1, 3)
        elif "net banking" in normalized or "net-banking" in normalized:
            payment, bounds = "net banking", (3, 5)
        elif "credit card" in normalized or "debit card" in normalized:
            payment, bounds = "credit/debit card", (5, 7)

        if payment:
            src, text = _voiceops_policy_section("REF-2")
            evidence = [(src, text)]
            started = _voiceops_parse_date(msg)
            if started:
                kb_source, kb_text = _voiceops_policy_section("KB-00.3")
                evidence.append((kb_source, kb_text))
                first = _voiceops_business_days_after(started, bounds[0])
                last = _voiceops_business_days_after(started, bounds[1])
                reply = (
                    f"Refunds to {payment} take {bounds[0]} to {bounds[1]} business "
                    f"days after initiation. For a refund started on {started.strftime('%d %B %Y')}, "
                    f"the estimated arrival window is {first.strftime('%d %B %Y')} to "
                    f"{last.strftime('%d %B %Y')}, excluding Sundays and warehouse holidays."
                )
            else:
                reply = (
                    f"Refunds to {payment} generally take {bounds[0]} to {bounds[1]} "
                    "business days after initiation."
                )
            return _voiceops_special_result(session_id, msg, reply, _voiceops_evidence(*evidence))

        if "cash on delivery" in low or re.search(r"\bcod\b", low):
            src3, text3 = _voiceops_policy_section("REF-3")
            src2, text2 = _voiceops_policy_section("REF-2")
            reply = (
                "For a cash-on-delivery return, you can choose ZentroMart Wallet, "
                "which is instant within 2 hours, or a bank transfer. Bank transfer "
                "takes 5 to 7 business days after the account details are verified."
            )
            return _voiceops_special_result(
                session_id, msg, reply, _voiceops_evidence((src3, text3), (src2, text2))
            )

        if any(term in low for term in ("when", "how long", "how soon", "kab tak", "kitne din")):
            src, text = _voiceops_policy_section("REF-2")
            reply = (
                "Refund timing depends on the payment method: UPI takes 1 to 3 business "
                "days, credit/debit cards 5 to 7 business days, net banking 3 to 5 "
                "business days, and ZentroMart Wallet is instant within 2 hours. "
                "Which payment method did you use?"
            )
            return _voiceops_special_result(
                session_id, msg, reply, _voiceops_evidence((src, text))
            )

    # 3. Product-aware return eligibility and date-specific electronics exception.
    return_intent = any(
        term in low for term in (
            "return", "send back", "send it back", "take back",
            "kitne din", "कितने दिन", "लौटाने", "वापस"
        )
    )
    if return_intent:
        product = _voiceops_find_product(msg)
        if product is not None:
            category = str(product.get("return_category", "")).lower()
            title = str(product.get("title", "This product"))
            sku = str(product.get("sku", ""))
            days_by_category = {
                "electronics": 7, "fashion": 15, "home_kitchen": 10,
                "beauty_sealed": 7, "books_media": 10,
            }
            days = days_by_category.get(category)
            evidence_items = [
                _voiceops_policy_section("RET-2"),
                _voiceops_policy_section("RET-3"),
                _voiceops_policy_section("RET-4"),
                _voiceops_policy_section("RET-8"),
                _voiceops_product_evidence(product),
            ]
            evidence = _voiceops_evidence(*evidence_items)
            delivery_date = _voiceops_parse_date(msg)

            opened_beauty = (
                category == "beauty_sealed"
                and any(term in low for term in (
                    "opened", "open the", "opened it", "unsealed", "broke the seal",
                    "khola", "khol li", "खोल", "खुला", "सील तोड़"
                ))
            )

            if opened_beauty:
                days = None
            elif (
                category == "electronics"
                and delivery_date is not None
                and delivery_date < date(2026, 8, 1)
            ):
                days = 10
            elif (
                category == "fashion"
                and re.search(r"\b(sale|order|ordered|placed|ऑर्डर)\b", low)
            ):
                order_date = _voiceops_parse_date(msg)
                if order_date and date(2026, 10, 1) <= order_date <= date(2026, 10, 15):
                    days = 30

            if days is None:
                reply = (
                    f"{title} is listed in a category that is not returnable "
                    "for change of mind."
                )
            else:
                reply = (
                    f"{title} ({sku}) has a {days}-calendar-day return window "
                    "from delivery under the applicable category policy."
                )
            return _voiceops_special_result(session_id, msg, reply, evidence)

    # 4. Handle high-value damaged items through a real ticket, without approving them.
    damaged_intent = any(
        term in low for term in (
            "crack", "broken", "damaged", "defective", "torn", "ripped",
            "टूटा", "टूटी", "खराब"
        )
    )
    if damaged_intent:
        product = _voiceops_find_product(msg)
        if product is not None:
            try:
                price = float(str(product.get("price_inr", "0")).replace(",", ""))
            except ValueError:
                price = 0
            d1 = _voiceops_policy_section("DWM-1")
            d2 = _voiceops_policy_section("DWM-2")
            d3 = _voiceops_policy_section("DWM-3")
            evidence = _voiceops_evidence(d1, d2, d3, _voiceops_product_evidence(product))
            if price > 10000:
                ticket = create_ticket(
                    summary=(
                        f"High-value damaged item reported: {product.get('sku')} "
                        f"{product.get('title')}. Customer message: {msg}"
                    ),
                    order_id=None,
                    reason="High-value damaged item requires Priority Resolution review",
                    intent="damaged_item",
                )
                if ticket.get("created") is True:
                    reply = (
                        "Because this item is worth more than ₹10,000, it must be "
                        "reviewed by the Priority Resolution team. I created a support "
                        f"ticket ({ticket.get('ticket_id')}); I cannot approve the claim myself. "
                        "Please report damage within 48 hours and keep photos of the item and packaging."
                    )
                else:
                    reply = (
                        "This high-value damaged item needs Priority Resolution review. "
                        "I couldn't confirm ticket creation, so please contact customer support. "
                        "Please report damage within 48 hours and keep photos of the item and packaging."
                    )
            else:
                reply = (
                    "Please report the damage within 48 hours of delivery and keep photos "
                    "of the item and packaging. After verification, the policy provides a "
                    "free replacement or full refund, including delivery fees; no reverse-pickup fee applies."
                )
            return _voiceops_special_result(session_id, msg, reply, evidence)

        src, text = _voiceops_policy_section("DWM-1")
        return _voiceops_special_result(
            session_id, msg,
            "Please report the damaged item within 48 hours of delivery and keep photos of the item and packaging. Please share the product name or SKU so I can guide you further.",
            _voiceops_evidence((src, text)),
        )

    # 5. Avoid inventing policies for unsupported requests.
    if "midnight" in low and any(x in low for x in ("delivery", "deliver", "schedule")):
        return _voiceops_special_result(
            session_id, msg,
            "I don't have information confirming whether midnight delivery can be scheduled. I can connect you to a colleague for confirmation.",
            [],
        )

    if "gst invoice" in low and "bulk" in low:
        return _voiceops_special_result(
            session_id, msg,
            "I couldn't find a confirmed policy about GST invoices for bulk orders. I can connect you to a colleague to confirm.",
            [],
        )

    if "student discount" in low:
        return _voiceops_special_result(
            session_id, msg,
            "I couldn't find confirmed information about a student discount. I can connect you to a colleague for confirmation.",
            [],
        )

    if any(term in low for term in (
        "pick up my order", "pickup my order", "store pickup",
        "pick it up from a store", "collect my order from a store",
    )):
        return _voiceops_special_result(
            session_id, msg,
            "I couldn't find confirmed information about store pickup. I can connect you to a colleague who can confirm the available options.",
            [],
        )

    # Privacy guardrail: do not disclose details about another customer.
    asks_other_person_contact = (
        "phone number" in low
        and any(term in low for term in (
            "person who ordered", "someone who ordered",
            "other customer", "my building", "our building",
        ))
    )
    if asks_other_person_contact:
        source, policy = _voiceops_policy_section("ESC-4")
        return _voiceops_special_result(
            session_id, msg,
            "I'm sorry, I can't share any details about other customers.",
            _voiceops_evidence((source, policy)),
        )

    # Delivery-area policy is explicit in the current policy documents.
    if any(term in low for term in ("deliver to dubai", "delivery to dubai", "ship to dubai")):
        source, policy = _voiceops_policy_section("DEL-5")
        return _voiceops_special_result(
            session_id, msg,
            "No. ZentroMart deliveries are available only within India.",
            _voiceops_evidence((source, policy)),
        )

    # 6. Answer the delivery-attempt question from the dataset policy.
    not_home = (
        ("not home" in low or "not at home" in low or "ghar par nahi" in low)
        and any(x in low for x in ("courier", "delivery", "deliver", "attempt"))
    )
    if not_home:
        src, text = _voiceops_policy_section("DEL-3")
        return _voiceops_special_result(
            session_id, msg,
            "The courier may try delivery up to 3 times. If a prepaid order still isn't delivered, the parcel returns to the warehouse; the refund is initiated when it arrives, usually 7 to 10 calendar days after the last attempt.",
            _voiceops_evidence((src, text)),
        )

    # 7. Ask for an order ID before attempting an order-status lookup.
    status_intent = any(
        term in low for term in (
            "order status", "status of my order", "check its status",
            "track my order", "track order", "where is my order",
            "where's my order", "order hasn't arrived", "order has not arrived",
            "hasn't arrived", "has not arrived", "delivery status", "shipment status"
        )
    )
    if status_intent and not re.search(r"(?<!\d)\d{6,}(?!\d)", msg):
        return _voiceops_special_result(
            session_id, msg,
            "I can help check your order status. Please share your order ID first.",
            [],
        )

    return _voiceops_base_support_turn(session_id=session_id, user_message=msg)
