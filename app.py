"""
=============================================================================
HYBRID TICKET MANAGEMENT SYSTEM — app.py
=============================================================================
Single execution-ready script combining all 4 pipeline stages, plus an
MCP server mode that exposes route_uncertain_ticket over stdio.

  Stage 1 — Data loading & TF-IDF preprocessing      (Task 1)
  Stage 2 — ML classification + uncertainty flagging  (Task 2)
  Stage 3 — Groq LLM fallback via MCP tool logic      (Task 3)
  Stage 4 — Evaluation metrics & confusion matrix     (Task 4)
  Stage 5 — MCP server exposing route_uncertain_ticket (Task 5)

Run pipeline : python app.py
Run as MCP   : python app.py --mcp
Env          : GROQ_API_KEY must be set before running either mode
=============================================================================
"""

import re
import os
import sys
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, APIError, AuthenticationError, RateLimitError

# NOTE: add GROQ_API_KEY using .env file in to your env before running the script.
load_dotenv()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# MCP 
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hybrid-ticket-system")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD        = 0.50
DEPARTMENTS      = ["Technical", "Billing", "Account"]
DEPARTMENTS_SORT = ["Account", "Billing", "Technical"]   # sorted for CM
GROQ_MODEL       = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a customer-support ticket classifier for a B2B SaaS company.
Your ONLY job is to read a support ticket and return a JSON object — nothing else.

Classification rules:
  Technical  → software bugs, errors, crashes, connectivity, login failures
  Billing    → charges, invoices, refunds, payments, pricing, promo codes
  Account    → profile changes, subscription changes, ownership, access management

You MUST respond with ONLY this JSON structure (no markdown, no extra text):
{
  "predicted_department": "<Technical | Billing | Account>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining the classification>"
}"""



# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — RAW DATASET & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

RAW_TICKETS = [
    {"ticket_id": "T001", "text": "My internet connection keeps dropping every 30 minutes!! Router shows Error#404.", "department": "Technical"},
    {"ticket_id": "T002", "text": "The mobile app crashes whenever I try to upload a file (>2MB). Running iOS 17.1.", "department": "Technical"},
    {"ticket_id": "T003", "text": "Password reset link expired after 10 mins. Still getting 500 Internal Server Error.", "department": "Technical"},
    {"ticket_id": "T004", "text": "Software update failed at 78%. Error code: 0xC1900101. Can't boot the system now!!", "department": "Technical"},
    {"ticket_id": "T005", "text": "I was charged $149.99 twice on 12/05/2024. Please refund the duplicate transaction.", "department": "Billing"},
    {"ticket_id": "T006", "text": "My invoice for March shows $89 but the agreed rate was $59/month. Please correct.", "department": "Billing"},
    {"ticket_id": "T007", "text": "Promo code SAVE20 didn't apply during checkout. Still billed full price of $120.", "department": "Billing"},
    {"ticket_id": "T008", "text": "Need GST invoice for payment of Rs.4999 made on 01-06-2024 for tax filing purposes.", "department": "Billing"},
    {"ticket_id": "T009", "text": "Please update my email from old@email.com to new@email.com on my account profile.", "department": "Account"},
    {"ticket_id": "T010", "text": "I want to cancel my Premium subscription and downgrade to the Free plan immediately.", "department": "Account"},
    {"ticket_id": "T011", "text": "My account got locked after 5 failed login attempts. Need it unlocked ASAP please.", "department": "Account"},
    {"ticket_id": "T012", "text": "How do I transfer account ownership to a new admin user? The current owner left.", "department": "Account"},
]

# Ambiguous tickets to trigger LLM fallback
NEW_TICKETS = [
    {"ticket_id": "NEW_01", "text": "I need help urgently please contact me as soon as possible.", "department": "Account"},
    {"ticket_id": "NEW_02", "text": "The system is not working properly and I am very unhappy.",  "department": "Technical"},
]


def preprocess_text(text: str) -> str:
    """Lowercase → remove digits → remove punctuation → collapse whitespace."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_dataset(raw_tickets: list) -> pd.DataFrame:
    df = pd.DataFrame(raw_tickets)
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    return df[["ticket_id", "text", "cleaned_text", "department"]]


def vectorise_tfidf(cleaned_texts: pd.Series):
    vectorizer = TfidfVectorizer(
        max_features=50, ngram_range=(1, 2),
        sublinear_tf=True, min_df=1, stop_words="english",
    )
    X = vectorizer.fit_transform(cleaned_texts).toarray()
    return X, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ML CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_reference_vectors(X: np.ndarray, y: np.ndarray) -> dict:
    return {
        dept: X[y == dept].mean(axis=0)
        for dept in DEPARTMENTS
    }


def classify_cosine(vec: np.ndarray, ref_vecs: dict) -> dict:
    scores   = {d: cosine_similarity(vec, ref_vecs[d]) for d in DEPARTMENTS}
    best     = max(scores, key=scores.get)
    score    = scores[best]
    uncertain = score < THRESHOLD
    return {"pred": best, "confidence": round(score, 4), "uncertain": uncertain}


def train_logistic(X: np.ndarray, y: np.ndarray):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    lr    = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
    lr.fit(X, y_enc)
    return lr, le


def classify_logistic(vec: np.ndarray, lr, le) -> dict:
    proba     = lr.predict_proba(vec.reshape(1, -1))[0]
    idx       = np.argmax(proba)
    best      = le.inverse_transform([idx])[0]
    score     = proba[idx]
    uncertain = score < THRESHOLD
    return {"pred": best, "confidence": round(float(score), 4), "uncertain": uncertain}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — GROQ LLM FALLBACK  (MCP tool logic)
# ─────────────────────────────────────────────────────────────────────────────

def route_uncertain_ticket(raw_text: str) -> dict:
    """
    MCP tool: route_uncertain_ticket
    Calls Groq LLM to classify a ticket that ML flagged as uncertain.
    Returns strict JSON: predicted_department, confidence, reasoning.
    """
    logger.info("LLM fallback triggered for: %.60s...", raw_text)

    try:
        client   = Groq()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=256,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Classify this support ticket:\n\n{raw_text}"},
            ],
        )
        raw     = response.choices[0].message.content.strip()
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed  = json.loads(cleaned)

        dept = parsed.get("predicted_department", "")
        if dept not in {"Technical", "Billing", "Account"}:
            raise ValueError(f"Invalid department from LLM: '{dept}'")

        return {
            "predicted_department": dept,
            "confidence"          : round(float(parsed.get("confidence", 0.0)), 4),
            "reasoning"           : str(parsed.get("reasoning", "")),
            "source"              : "LLM Fallback (Groq)",
        }

    except AuthenticationError:
        logger.error("Groq auth failed — check GROQ_API_KEY")
        return {"predicted_department": "Unknown", "confidence": 0.0,
                "reasoning": "Auth error", "source": "LLM Error"}
    except RateLimitError:
        logger.error("Groq rate limit hit")
        return {"predicted_department": "Unknown", "confidence": 0.0,
                "reasoning": "Rate limit", "source": "LLM Error"}
    except (APIError, json.JSONDecodeError, ValueError) as exc:
        logger.error("LLM error: %s", exc)
        return {"predicted_department": "Unknown", "confidence": 0.0,
                "reasoning": str(exc), "source": "LLM Error"}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(hybrid_df: pd.DataFrame) -> None:
    y_true = hybrid_df["true_dept"].tolist()
    y_pred = hybrid_df["final_pred"].tolist()

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=["Technical"], average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, labels=["Technical"], average="macro", zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=DEPARTMENTS_SORT)
    report    = classification_report(y_true, y_pred, labels=DEPARTMENTS_SORT, zero_division=0)

    print_section("EVALUATION METRICS")
    bar = "█" * int(acc * 30)
    print(f"\n  Overall Accuracy       : {acc:.4f}  ({acc*100:.1f}%)  {bar}")
    print(f"  Technical — Precision  : {precision:.4f}")
    print(f"  Technical — Recall     : {recall:.4f}")

    print_section("CONFUSION MATRIX")
    col_w = 12
    print(f"\n  Rows = Actual  |  Columns = Predicted\n")
    label  = "Actual/Pred"
    header = f"  {label:<14}" + "".join(f"{d:>{col_w}}" for d in DEPARTMENTS_SORT)
    print(header)
    print("  " + "─" * (14 + col_w * len(DEPARTMENTS_SORT)))
    for i, dept in enumerate(DEPARTMENTS_SORT):
        row = f"  {dept:<14}"
        for j, val in enumerate(cm[i]):
            cell = f"[{val}]" if i == j else f" {val} "
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"\n  Misclassifications:")
    found = False
    for i, actual in enumerate(DEPARTMENTS_SORT):
        for j, pred in enumerate(DEPARTMENTS_SORT):
            if i != j and cm[i][j] > 0:
                print(f"    {cm[i][j]}x '{actual}' predicted as '{pred}'")
                found = True
    if not found:
        print(f"    None — perfect classification!")

    print_section("CLASSIFICATION REPORT")
    print()
    for line in report.splitlines():
        print(f"  {line}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — MCP SERVER (exposes route_uncertain_ticket over stdio)
# ─────────────────────────────────────────────────────────────────────────────

mcp_server = Server("ticket-routing-mcp")


@mcp_server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="route_uncertain_ticket",
            description=(
                "Classifies a low-confidence support ticket using Groq LLM. "
                "Returns JSON with predicted_department, confidence, and reasoning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "raw_text": {
                        "type": "string",
                        "description": "Raw customer support ticket text.",
                    }
                },
                "required": ["raw_text"],
            },
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "route_uncertain_ticket":
        return [types.TextContent(type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}))]

    raw_text = arguments.get("raw_text", "").strip()
    if not raw_text:
        return [types.TextContent(type="text",
                text=json.dumps({"error": "raw_text must be a non-empty string."}))]

    # route_uncertain_ticket() is synchronous (Groq SDK call) — run it off
    # the event loop so it doesn't block other MCP requests.
    result = await asyncio.to_thread(route_uncertain_ticket, raw_text)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def run_mcp_server() -> None:
    logger.info("Starting MCP server 'ticket-routing-mcp' on stdio...")
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    w = 70
    print(f"\n{'═' * w}\n  {title}\n{'═' * w}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 70)
    print("  HYBRID TICKET MANAGEMENT SYSTEM — FULL PIPELINE")
    print("═" * 70)

    # ── STAGE 1: Data Preparation ────────────────────────────────────────────
    print_section("STAGE 1 — DATA PREPARATION & TF-IDF VECTORISATION")
    df  = build_dataset(RAW_TICKETS)
    X, vectorizer = vectorise_tfidf(df["cleaned_text"])
    y   = df["department"].values
    print(f"\n  Tickets loaded   : {len(df)}")
    print(f"  TF-IDF matrix    : {X.shape}  (tickets × features)")
    print(f"  Departments      : {list(np.unique(y))}")

    # ── STAGE 2: ML Classification ───────────────────────────────────────────
    print_section("STAGE 2 — ML CLASSIFICATION (Cosine + Logistic Regression)")
    ref_vecs  = build_reference_vectors(X, y)
    lr, le    = train_logistic(X, y)

    results = []
    for i, row in df.iterrows():
        vec     = X[i]
        cos_res = classify_cosine(vec, ref_vecs)
        lr_res  = classify_logistic(vec, lr, le)
        uncertain = cos_res["uncertain"] or lr_res["uncertain"]
        results.append({
            "ticket_id"   : row["ticket_id"],
            "true_dept"   : row["department"],
            "raw_text"    : row["text"],
            "cos_pred"    : cos_res["pred"],
            "cos_conf"    : cos_res["confidence"],
            "lr_pred"     : lr_res["pred"],
            "lr_conf"     : lr_res["confidence"],
            "uncertain"   : uncertain,
        })

    ml_df = pd.DataFrame(results)
    print(f"\n  {'ID':<8} {'True':<12} {'LR Pred':<12} {'Conf':<8} {'Uncertain'}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*8} {'─'*10}")
    for _, r in ml_df.iterrows():
        flag = "⚠ YES → LLM" if r["uncertain"] else "✓ NO"
        print(f"  {r['ticket_id']:<8} {r['true_dept']:<12} {r['lr_pred']:<12} {r['lr_conf']:<8.4f} {flag}")

    # ── STAGE 3: Groq LLM Fallback ───────────────────────────────────────────
    print_section("STAGE 3 — GROQ LLM FALLBACK (MCP tool: route_uncertain_ticket)")

    # Run on training tickets flagged uncertain + new ambiguous tickets
    uncertain_tickets = ml_df[ml_df["uncertain"]].to_dict("records")
    for nt in NEW_TICKETS:
        nt["cleaned"] = preprocess_text(nt["text"])
        vec           = vectorizer.transform([nt["cleaned"]]).toarray()[0]
        lr_res        = classify_logistic(vec, lr, le)
        uncertain_tickets.append({
            "ticket_id": nt["ticket_id"],
            "true_dept": nt["department"],
            "raw_text" : nt["text"],
            "uncertain": lr_res["uncertain"] or lr_res["confidence"] < THRESHOLD,
        })

    llm_results = {}
    if uncertain_tickets:
        print(f"\n  {len(uncertain_tickets)} ticket(s) routed to LLM fallback:\n")
        for t in uncertain_tickets:
            result = route_uncertain_ticket(t["raw_text"])
            llm_results[t["ticket_id"]] = result
            print(f"  [{t['ticket_id']}] → {result['predicted_department']}"
                  f"  (confidence: {result['confidence']})  |  {result['reasoning']}")
    else:
        print("\n  No uncertain tickets — LLM fallback not triggered.")

    # ── STAGE 4: Build hybrid df & evaluate ──────────────────────────────────
    print_section("STAGE 4 — EVALUATION METRICS")

    hybrid_rows = []
    for _, r in ml_df.iterrows():
        if r["uncertain"] and r["ticket_id"] in llm_results:
            final = llm_results[r["ticket_id"]]["predicted_department"]
            source = "LLM Fallback (Groq)"
        else:
            final  = r["lr_pred"]
            source = "ML (Logistic Regression)"
        hybrid_rows.append({
            "ticket_id" : r["ticket_id"],
            "true_dept" : r["true_dept"],
            "final_pred": final,
            "source"    : source,
        })

    hybrid_df = pd.DataFrame(hybrid_rows)
    evaluate(hybrid_df)

    # ── Export ────────────────────────────────────────────────────────────────
    print_section("EXPORT")
    df.to_csv("tickets_cleaned.csv", index=False)
    ml_df.to_csv("ml_results.csv", index=False)
    hybrid_df.to_csv("hybrid_evaluation.csv", index=False)
    print(f"\n  ✓ tickets_cleaned.csv")
    print(f"  ✓ ml_results.csv")
    print(f"  ✓ hybrid_evaluation.csv")

    print(f"\n{'═'*70}")
    print(f"  PIPELINE COMPLETE: Stage 1 → Stage 2 → Stage 3 → Stage 4  ✓")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    if "--mcp" in sys.argv:
        asyncio.run(run_mcp_server())
    else:
        main()


















