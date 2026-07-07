"""
=============================================================================
TICKET ROUTING MCP SERVER — server.py
=============================================================================
A standalone, long-running MCP server. Start it once, independently of any
client:

    python server.py                          # streamable HTTP on :8765
    python server.py --transport stdio         # stdio, for local dev/tests

main.py connects to whichever instance is already running — it never
spawns this process itself.

Exposes:
  Tools
    - route_uncertain_ticket : classifies a low-confidence ticket
                                (the only tool main.py's pipeline calls)
    - summarize_ticket       : one-line ticket summary — implemented and
                                listed, reserved for a future feature, not
                                called anywhere in the current pipeline
  Resources
    - taxonomy://departments : the department definitions used for
                                classification, readable independently of
                                any tool call
  Prompts
    - classify_ticket        : the reusable classification prompt template
    - summarize_ticket       : the reusable summarization prompt template

LLM backend is pluggable via LLM_PROVIDER (groq | openai | ollama | claude).
Tool logic below only ever calls LLMProvider.complete() — it never imports
an SDK directly, so swapping providers never touches tool code.
=============================================================================
"""

import argparse
import json
import logging
import os
import re
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ticket-routing-mcp-server")


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROVIDERS — pluggable backend, selected by LLM_PROVIDER env var
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
    "claude": "claude-haiku-4-5-20251001",
}


class LLMProvider(ABC):
    """A chat-completion backend that tools call through, never directly."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion for a system/user prompt pair."""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        from groq import Groq
        self._client = Groq()
        self._model = model or os.getenv("LLM_MODEL", DEFAULT_MODELS["groq"])

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        import asyncio

        def _call():
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        return await asyncio.to_thread(_call)


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI()
        self._model = model or os.getenv("LLM_MODEL", DEFAULT_MODELS["openai"])

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=256,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server — no API key needed."""

    def __init__(self, model: str | None = None, host: str | None = None):
        import ollama
        self._client = ollama.AsyncClient(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self._model = model or os.getenv("LLM_MODEL", DEFAULT_MODELS["ollama"])

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat(
            model=self._model,
            options={"temperature": 0.0},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic()
        self._model = model or os.getenv("LLM_MODEL", DEFAULT_MODELS["claude"])

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


def get_llm_provider() -> LLMProvider:
    """Single switch for which LLM backs this server. Everything below is
    unaware of the choice — it only ever calls LLMProvider.complete()."""
    name = os.getenv("LLM_PROVIDER", "groq").lower()
    if name == "groq":
        return GroqProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "ollama":
        return OllamaProvider()
    if name in ("claude", "anthropic"):
        return ClaudeProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Supported: groq, openai, ollama, claude")


# ─────────────────────────────────────────────────────────────────────────────
# DEPARTMENT TAXONOMY — shared by the classify tool, the resource, and the prompt
# ─────────────────────────────────────────────────────────────────────────────

DEPARTMENT_TAXONOMY = {
    "Technical": "software bugs, errors, crashes, connectivity, login failures",
    "Billing": "charges, invoices, refunds, payments, pricing, promo codes",
    "Account": "profile changes, subscription changes, ownership, access management",
}
VALID_DEPARTMENTS = set(DEPARTMENT_TAXONOMY)


def department_taxonomy_text() -> str:
    return "\n".join(f"{dept}: {desc}" for dept, desc in DEPARTMENT_TAXONOMY.items())


def _strip_json_fence(raw: str) -> str:
    return re.sub(r"```(?:json)?|```", "", raw).strip()


# ─────────────────────────────────────────────────────────────────────────────
# TOOL LOGIC — pure functions over an LLMProvider, registered as MCP tools below
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = f"""You are a customer-support ticket classifier for a B2B SaaS company.
Your ONLY job is to read a support ticket and return a JSON object — nothing else.

Department definitions:
{department_taxonomy_text()}

You MUST respond with ONLY this JSON structure (no markdown, no extra text):
{{
  "predicted_department": "<Technical | Billing | Account>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining the classification>"
}}"""

SUMMARIZE_SYSTEM_PROMPT = """You summarize customer-support tickets for a B2B SaaS company.
Read the ticket and return ONLY this JSON structure (no markdown, no extra text):
{
  "summary": "<one sentence, under 20 words, capturing the core issue>"
}"""


async def route_uncertain_ticket_logic(raw_text: str, llm: LLMProvider) -> dict:
    """Classifies a ticket the ML stage flagged as low-confidence."""
    logger.info("route_uncertain_ticket: %.60s...", raw_text)
    try:
        raw = await llm.complete(CLASSIFY_SYSTEM_PROMPT, f"Classify this support ticket:\n\n{raw_text}")
        parsed = json.loads(_strip_json_fence(raw))

        dept = parsed.get("predicted_department", "")
        if dept not in VALID_DEPARTMENTS:
            raise ValueError(f"Invalid department from LLM: '{dept}'")

        return {
            "predicted_department": dept,
            "confidence": round(float(parsed.get("confidence", 0.0)), 4),
            "reasoning": str(parsed.get("reasoning", "")),
            "source": "LLM Fallback",
        }
    except Exception as exc:  # provider-agnostic: any SDK can raise here
        logger.error("route_uncertain_ticket failed: %s", exc)
        return {"predicted_department": "Unknown", "confidence": 0.0,
                "reasoning": str(exc), "source": "LLM Error"}


async def summarize_ticket_logic(raw_text: str, llm: LLMProvider) -> dict:
    """One-line ticket summary. Not called by the current pipeline — reserved
    for a future feature (e.g. an agent-facing ticket preview)."""
    logger.info("summarize_ticket: %.60s...", raw_text)
    try:
        raw = await llm.complete(SUMMARIZE_SYSTEM_PROMPT, f"Summarize this support ticket:\n\n{raw_text}")
        parsed = json.loads(_strip_json_fence(raw))
        return {"summary": str(parsed.get("summary", "")).strip(), "source": "LLM Summary"}
    except Exception as exc:  # provider-agnostic: any SDK can raise here
        logger.error("summarize_ticket failed: %s", exc)
        return {"summary": "", "source": "LLM Error", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# MCP SERVER — tools, resource, prompts
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("ticket-routing-mcp")
llm = get_llm_provider()


@mcp.tool(
    name="route_uncertain_ticket",
    description=(
        "Classifies a low-confidence support ticket using the configured LLM "
        "provider. Returns predicted_department, confidence, and reasoning."
    ),
)
async def route_uncertain_ticket_tool(raw_text: str) -> dict:
    return await route_uncertain_ticket_logic(raw_text, llm)


@mcp.tool(
    name="summarize_ticket",
    description=(
        "Summarizes a support ticket into one sentence. Reserved for future "
        "use — not called by the current routing pipeline."
    ),
)
async def summarize_ticket_tool(raw_text: str) -> dict:
    return await summarize_ticket_logic(raw_text, llm)


@mcp.resource("taxonomy://departments")
def department_taxonomy() -> str:
    """Department definitions used for ticket classification."""
    return department_taxonomy_text()


@mcp.prompt(name="classify_ticket")
def classify_ticket_prompt(ticket_text: str) -> str:
    """Reusable prompt template behind the route_uncertain_ticket tool."""
    return f"{CLASSIFY_SYSTEM_PROMPT}\n\nClassify this support ticket:\n\n{ticket_text}"


@mcp.prompt(name="summarize_ticket")
def summarize_ticket_prompt(ticket_text: str) -> str:
    """Reusable prompt template behind the summarize_ticket tool."""
    return f"{SUMMARIZE_SYSTEM_PROMPT}\n\nSummarize this support ticket:\n\n{ticket_text}"


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket routing MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="streamable-http")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    logger.info("LLM provider: %s", llm.__class__.__name__)
    logger.info("Starting ticket-routing-mcp (transport=%s)", args.transport)
    if args.transport == "streamable-http":
        logger.info("Listening on http://%s:%s%s", args.host, args.port, mcp.settings.streamable_http_path)

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
