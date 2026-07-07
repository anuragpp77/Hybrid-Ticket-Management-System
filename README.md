# Hybrid Ticket Management System

A support-ticket classification pipeline that combines classic ML (TF-IDF +
Logistic Regression / cosine similarity) with an **LLM fallback for
low-confidence tickets**, served over the **Model Context Protocol (MCP)**.

The system is split into two independent processes:

| File | Role |
|---|---|
| `server.py` | Long-running MCP server. Wraps an LLM provider (Groq / OpenAI / Ollama / Claude) behind MCP tools, a resource, and prompts. |
| `main.py` | The pipeline. Loads data, runs ML classification, and — only for tickets the ML stage is unsure about — calls the MCP server to get an LLM-based classification. |

`main.py` **never** imports an LLM SDK and **never** spawns the server; it only
talks to a server that is already running, via an MCP client.

---

## Architecture

```
┌─────────────────────────────┐         MCP (HTTP or stdio)        ┌──────────────────────────────┐
│           main.py           │ ──────────────────────────────▶     │           server.py            │
│                             │                                      │                                │
│ Stage 1  Data + TF-IDF      │                                      │  MCP tools:                    │
│ Stage 2  ML classification  │                                      │   - route_uncertain_ticket      │
│          (cosine + logistic)│                                      │   - summarize_ticket (reserved) │
│ Stage 3  LLM fallback ───────▶│  route_uncertain_ticket(text)      │                                │
│          (via MCP client)   │◀───────────────────────────────     │  MCP resource:                  │
│ Stage 4  Evaluation metrics │                                      │   - taxonomy://departments       │
│          + CSV export       │                                      │                                │
└─────────────────────────────┘                                      │  MCP prompts:                    │
                                                                     │   - classify_ticket              │
                                                                     │   - summarize_ticket             │
                                                                     │                                │
                                                                     │  LLMProvider.complete() ────────▶│──▶ Groq / OpenAI / Ollama / Claude
                                                                     └──────────────────────────────┘
```

**Key point:** MCP is only the transport between `main.py` and `server.py`.
The LLM itself is a plain chat-completion call made *inside* the server's
tool handler — the model has no awareness of MCP. Anything the model needs
(like the department taxonomy) is baked into its system prompt as plain text
ahead of time.

---

## Requirements

```bash
pip install mcp python-dotenv scikit-learn pandas numpy
```

Plus whichever LLM SDK matches your chosen provider:

```bash
pip install groq          # LLM_PROVIDER=groq   (default)
pip install openai        # LLM_PROVIDER=openai
pip install ollama        # LLM_PROVIDER=ollama
pip install anthropic     # LLM_PROVIDER=claude
```

## Configuration

Create a `.env` file (loaded automatically via `python-dotenv`):

```bash
# Which backend the server uses. One of: groq | openai | ollama | claude
LLM_PROVIDER=groq

# Optional: override the default model for the chosen provider
LLM_MODEL=llama-3.1-8b-instant

# Required by whichever SDK you use (Groq/OpenAI/Anthropic pick these up automatically)
GROQ_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Only needed for Ollama, if not running on localhost
OLLAMA_HOST=http://localhost:11434

# main.py: where to find the running MCP server
MCP_SERVER_URL=http://127.0.0.1:8765/mcp
```

Default models per provider (used if `LLM_MODEL` isn't set):

| Provider | Default model |
|---|---|
| groq | `llama-3.1-8b-instant` |
| openai | `gpt-4o-mini` |
| ollama | `llama3.1` |
| claude | `claude-haiku-4-5-20251001` |

## Running

**1. Start the MCP server first, in its own terminal.** It stays running
independently of any pipeline run and can serve multiple clients/requests.

```bash
python server.py                          # streamable HTTP on :8765 (default)
python server.py --transport stdio        # stdio transport, for local dev
python server.py --host 0.0.0.0 --port 9000
```

**2. Run the pipeline** in a second terminal:

```bash
python main.py
```

If the server isn't reachable, `main.py` prints an error and exits rather
than trying to start one itself.

---

## Pipeline stages (`main.py`)

1. **Data preparation & TF-IDF** — loads a small hardcoded set of labeled
   tickets (`RAW_TICKETS`), cleans the text (lowercase, strip digits/punct),
   and vectorizes it with `TfidfVectorizer` (unigrams+bigrams, 50 features).
2. **ML classification** — classifies every ticket two ways: cosine
   similarity against per-department mean vectors, and a `LogisticRegression`
   model. A ticket is flagged **uncertain** if either method's confidence
   falls below `THRESHOLD = 0.50`.
3. **LLM fallback via MCP** — every uncertain ticket (plus a small hardcoded
   batch of intentionally ambiguous `NEW_TICKETS`) is sent to the running MCP
   server's `route_uncertain_ticket` tool through `TicketRoutingClient`. The
   server's LLM returns a predicted department, confidence, and one-line
   reasoning.
4. **Evaluation** — merges ML predictions and LLM fallback predictions into
   one hybrid result set, then prints accuracy, precision/recall (Technical),
   a confusion matrix, and a full classification report.

Outputs written to disk:

- `tickets_cleaned.csv` — original + cleaned ticket text
- `ml_results.csv` — per-ticket ML predictions/confidence/uncertainty flags
- `hybrid_evaluation.csv` — final predictions (ML or LLM) vs. ground truth

---

## MCP server (`server.py`)

### Tools

| Tool | Description |
|---|---|
| `route_uncertain_ticket` | Classifies a low-confidence ticket into `Technical`, `Billing`, or `Account`. Returns `predicted_department`, `confidence`, `reasoning`, `source`. The only tool the current pipeline calls. |
| `summarize_ticket` | Produces a one-sentence ticket summary. Implemented and listed, but not called anywhere in the current pipeline — reserved for a future feature. |

### Resource

| Resource | Description |
|---|---|
| `taxonomy://departments` | The department definitions used for classification, readable independently of any tool call. |

### Prompts

| Prompt | Description |
|---|---|
| `classify_ticket` | Reusable prompt template backing `route_uncertain_ticket`. |
| `summarize_ticket` | Reusable prompt template backing `summarize_ticket`. |

### LLM providers

All providers implement the same `LLMProvider.complete(system_prompt, user_prompt) -> str`
interface, so tool logic never touches an SDK directly — swapping
`LLM_PROVIDER` is the only thing that changes.

- `GroqProvider` — Groq's OpenAI-compatible chat API
- `OpenAIProvider` — OpenAI chat completions (async client)
- `OllamaProvider` — local Ollama server, no API key required
- `ClaudeProvider` — Anthropic Messages API

Any failure inside a provider call (bad JSON, API error, invalid department
in the response) is caught and returned as a safe fallback result:
`{"predicted_department": "Unknown", "confidence": 0.0, "source": "LLM Error", ...}`,
so the pipeline never crashes on a bad LLM response.

---

## `TicketRoutingClient` (MCP client, in `main.py`)

An async context manager around an MCP `ClientSession`:

```python
async with TicketRoutingClient() as client:
    result = await client.route_uncertain_ticket("My invoice looks wrong")
```

- Defaults to connecting over **streamable HTTP** to `MCP_SERVER_URL`.
- `TicketRoutingClient.for_stdio(command, args)` is an alternate constructor,
  for local dev, that spawns the server as a subprocess and talks to it over
  stdio instead of connecting to an already-running instance.
- Uses `AsyncExitStack` to guarantee the transport and session are cleaned
  up correctly, in order, on exit — even if something fails mid-setup.
- `_call_tool` normalizes MCP's two possible result shapes (`structuredContent`
  vs. a text content block containing JSON) into a plain `dict`.
- `list_tools()` and `get_department_taxonomy()` are introspection helpers,
  available but unused by the current pipeline.

---

## Notes / limitations

- The dataset is a small hardcoded list for demonstration — swap
  `RAW_TICKETS` / `NEW_TICKETS` for a real data source in production.
- `THRESHOLD = 0.50` controls how aggressively tickets are routed to the LLM;
  lower it to rely more on ML, raise it to lean more on the LLM.
- `summarize_ticket` and `taxonomy://departments` are fully working but
  unused by the current pipeline — hooks for future features (e.g. an
  agent-facing ticket preview).