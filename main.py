"""
=============================================================================
HYBRID TICKET MANAGEMENT SYSTEM — main.py
=============================================================================
Runs the full pipeline. Stages 1, 2, and 4 execute in-process here. Stage 3
(LLM fallback for uncertain tickets) is handled by a separately-running MCP
server (server.py) — this file only ever talks to it through the
TicketRoutingClient below; it never spawns the server and never imports an
LLM SDK.

Before running this, start the server once, separately:
    python server.py

Then run the pipeline:
    python main.py

  Stage 1 — Data loading & TF-IDF preprocessing
  Stage 2 — ML classification + uncertainty flagging
  Stage 3 — LLM fallback via MCP client → running MCP server
  Stage 4 — Evaluation metrics & confusion matrix
=============================================================================
"""

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack

import numpy as np
import pandas as pd
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD        = 0.50
DEPARTMENTS      = ["Technical", "Billing", "Account"]
DEPARTMENTS_SORT = ["Account", "Billing", "Technical"]   # sorted for CM

DEFAULT_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8765/mcp")

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
    scores    = {d: cosine_similarity(vec, ref_vecs[d]) for d in DEPARTMENTS}
    best      = max(scores, key=scores.get)
    score     = scores[best]
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
# STAGE 3 — MCP CLIENT (talks to the already-running server, never spawns it)
# ─────────────────────────────────────────────────────────────────────────────

class TicketRoutingClient:
    """Connects to a ticket-routing-mcp server that is ALREADY RUNNING
    elsewhere (started via `python server.py`). This client never spawns a
    server process — one server can serve many clients/requests, and its
    lifetime is independent of any single pipeline run.

        async with TicketRoutingClient() as client:
            result = await client.route_uncertain_ticket("some ticket text")

    For local development only, `TicketRoutingClient.for_stdio(...)` spawns
    a server subprocess directly over stdio instead of connecting over HTTP.
    """

    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self._server_url = server_url
        self._stdio_params: StdioServerParameters | None = None
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    @classmethod
    def for_stdio(cls, command: str, args: list[str]) -> "TicketRoutingClient":
        instance = cls(server_url="")
        instance._stdio_params = StdioServerParameters(command=command, args=args)
        return instance

    async def __aenter__(self) -> "TicketRoutingClient":
        if self._stdio_params is not None:
            read_stream, write_stream = await self._stack.enter_async_context(
                stdio_client(self._stdio_params)
            )
        else:
            read_stream, write_stream, _ = await self._stack.enter_async_context(
                streamablehttp_client(self._server_url)
            )
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._stack.aclose()

    async def _call_tool(self, tool_name: str, **arguments) -> dict:
        result = await self._session.call_tool(tool_name, arguments)
        if result.structuredContent is not None:
            return result.structuredContent
        return json.loads(result.content[0].text)

    async def route_uncertain_ticket(self, raw_text: str) -> dict:
        """Calls the 'route_uncertain_ticket' tool. The only tool this
        pipeline actually invokes."""
        return await self._call_tool("route_uncertain_ticket", raw_text=raw_text)

    async def summarize_ticket(self, raw_text: str) -> dict:
        """Calls the 'summarize_ticket' tool. Not used by the current
        pipeline — available for a future feature."""
        return await self._call_tool("summarize_ticket", raw_text=raw_text)

    async def list_tools(self) -> list[str]:
        result = await self._session.list_tools()
        return [t.name for t in result.tools]

    async def get_department_taxonomy(self) -> str:
        """Reads the 'taxonomy://departments' resource from the server."""
        result = await self._session.read_resource("taxonomy://departments")
        return result.contents[0].text


async def run_llm_fallback(uncertain_tickets: list, client: TicketRoutingClient) -> dict:
    """Routes each uncertain ticket to the server's route_uncertain_ticket
    tool and returns {ticket_id: result}."""
    llm_results = {}
    for t in uncertain_tickets:
        result = await client.route_uncertain_ticket(t["raw_text"])
        llm_results[t["ticket_id"]] = result
        print(f"  [{t['ticket_id']}] → {result['predicted_department']}"
              f"  (confidence: {result['confidence']})  |  {result['reasoning']}")
    return llm_results


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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    w = 70
    print(f"\n{'═' * w}\n  {title}\n{'═' * w}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def main():
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
    ref_vecs = build_reference_vectors(X, y)
    lr, le   = train_logistic(X, y)

    results = []
    for i, row in df.iterrows():
        vec       = X[i]
        cos_res   = classify_cosine(vec, ref_vecs)
        lr_res    = classify_logistic(vec, lr, le)
        uncertain = cos_res["uncertain"] or lr_res["uncertain"]
        results.append({
            "ticket_id" : row["ticket_id"],
            "true_dept" : row["department"],
            "raw_text"  : row["text"],
            "cos_pred"  : cos_res["pred"],
            "cos_conf"  : cos_res["confidence"],
            "lr_pred"   : lr_res["pred"],
            "lr_conf"   : lr_res["confidence"],
            "uncertain" : uncertain,
        })

    ml_df = pd.DataFrame(results)
    print(f"\n  {'ID':<8} {'True':<12} {'LR Pred':<12} {'Conf':<8} {'Uncertain'}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*8} {'─'*10}")
    for _, r in ml_df.iterrows():
        flag = "⚠ YES → LLM" if r["uncertain"] else "✓ NO"
        print(f"  {r['ticket_id']:<8} {r['true_dept']:<12} {r['lr_pred']:<12} {r['lr_conf']:<8.4f} {flag}")

    # ── STAGE 3: LLM Fallback via MCP (server must already be running) ──────
    print_section("STAGE 3 — LLM FALLBACK (MCP client → running MCP server)")

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
        print(f"\n  {len(uncertain_tickets)} ticket(s) routed to LLM fallback")
        print(f"  Connecting to MCP server at {DEFAULT_SERVER_URL} ...\n")
        try:
            async with TicketRoutingClient() as client:
                llm_results = await run_llm_fallback(uncertain_tickets, client)
        except OSError as exc:
            print(f"\n  ✗ Could not reach the MCP server: {exc}")
            print(f"  Start it first in another terminal:  python server.py")
            sys.exit(1)
    else:
        print("\n  No uncertain tickets — LLM fallback not triggered.")

    # ── STAGE 4: Build hybrid df & evaluate ──────────────────────────────────
    print_section("STAGE 4 — EVALUATION METRICS")

    hybrid_rows = []
    for _, r in ml_df.iterrows():
        if r["uncertain"] and r["ticket_id"] in llm_results:
            final  = llm_results[r["ticket_id"]]["predicted_department"]
            source = "LLM Fallback"
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
    asyncio.run(main())
