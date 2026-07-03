# Hybrid Ticket Management System

An intelligent customer support ticket routing system that combines traditional Machine Learning with an LLM fallback via an MCP server to classify tickets into **Technical**, **Billing**, or **Account** departments.

---

## Project Structure

```
├── app.py                         # Single execution-ready pipeline (all stages)
├── task1.py                       # Stage 1: Data preparation & TF-IDF
├── task2.py                       # Stage 2: ML classification
├── task3.py                       # Stage 3: MCP server + Groq LLM fallback
├── task4.py                       # Stage 4: Evaluation metrics
├── grok_text.py                   # Standalone Groq LLM fallback test
├── requirements.txt               # Python dependencies with versions
└── README.md                      # This file
```

---

## Environment Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

Get your free API key from [console.groq.com](https://console.groq.com) → API Keys → Create new key.

then add .env file with key into the environment
```

---

## Running the System

### Run the full pipeline (recommended)
```bash
python app.py
```
This executes all 4 stages in sequence and exports results as CSV files.

### Run individual stages
```bash
python task1.py    # Data preparation
python task2.py    # ML classification
python task3.py    # MCP server (stays running — waits for tool calls)
python task4.py    # Evaluation metrics
```


---

## Methodology

### Stage 1 — Data Preparation
Raw support tickets are cleaned using a custom text preprocessor (lowercase, remove digits and punctuation, collapse whitespace) and transformed into a numerical TF-IDF feature matrix using scikit-learn's `TfidfVectorizer` with unigrams and bigrams.

### Stage 2 — ML Classification
Two complementary classifiers run on every ticket:

- **Cosine Similarity**: Each ticket vector is compared against per-department centroid vectors (average of all training vectors for that department). The department with the highest cosine score wins.
- **Logistic Regression**: A multi-class LR model trained on the TF-IDF matrix produces calibrated class probabilities via `predict_proba`.

If either classifier returns a confidence score below the **0.50 threshold**, the ticket is flagged as `UNCERTAIN` and escalated to the LLM fallback.

### Stage 3 — MCP Server + LLM Fallback
An MCP server exposes a single tool: `route_uncertain_ticket`. When called with a raw ticket string, it sends the text to the **Groq API** (model: `llama-3.1-8b-instant`) with a strict system prompt that forces a JSON response containing:
- `predicted_department` — must be Technical, Billing, or Account
- `confidence` — float between 0 and 1
- `reasoning` — one-sentence explanation

The server handles and logs all error types transparently (auth errors, rate limits, malformed JSON).

### Stage 4 — Evaluation Metrics
The hybrid system (ML predictions + LLM overrides for uncertain tickets) is evaluated using:
- **Overall Accuracy** — proportion of correctly classified tickets
- **Precision (Technical)** — of all tickets predicted as Technical, how many truly are
- **Recall (Technical)** — of all actual Technical tickets, how many were correctly identified
- **Confusion Matrix** — text-based 3×3 grid showing misclassifications per department
- **Full Classification Report** — per-class precision, recall, F1, and support

---

## Output Files

| File | Description |
|---|---|
| `tickets_cleaned.csv` | Preprocessed dataset with cleaned text |
| `tfidf_features.csv` | TF-IDF feature matrix |
| `ml_results.csv` | Per-ticket ML predictions and confidence scores |
| `hybrid_evaluation.csv` | Final hybrid predictions (ML + LLM) |
| `task4_metrics_summary.txt` | Accuracy, precision, recall summary |
