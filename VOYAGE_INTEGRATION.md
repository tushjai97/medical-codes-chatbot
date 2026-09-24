# Voyage-4-lite Embedding + Local Reranker Integration

This branch (`voyage-4-lite-embeddings`) swaps the pipeline's embedding
model from local `sentence-transformers/all-MiniLM-L6-v2` (384-dim) to
Voyage AI's hosted `voyage-4-lite` (1024-dim), replaces the Perplexity
LLM reranker (Expert mode) with a local open-source cross-encoder
(`BAAI/bge-reranker-v2-m3`), and adds tooling to spot-check and measure
retrieval performance.

## Pipeline structure

```
medical-codes-chatbot/
├── backend/
│   ├── app/
│   │   ├── config.py              # Settings: VOYAGE_API_KEY, VOYAGE_MODEL_NAME, EMBEDDING_DIM
│   │   ├── main.py                # FastAPI app, /api/code-suggestions endpoint
│   │   ├── database.py            # Postgres connection pool
│   │   └── services/
│   │       ├── embeddings.py      # EmbeddingService — calls Voyage's embed API
│   │       ├── vector_search.py   # pgvector cosine similarity search (<=> operator)
│   │       ├── keyword_search.py  # Postgres full-text search
│   │       ├── hybrid_search.py   # Combines vector + keyword via RRF (search_all())
│   │       ├── ranking.py         # Reciprocal Rank Fusion + score normalization
│   │       └── reranker_service.py # BAAI/bge-reranker-v2-m3 cross-encoder (Expert mode only)
│   └── scripts/
│       ├── setup_database.py      # Creates cpt_codes/icd10_codes tables + indexes
│       ├── load_cpt_codes.py      # Embeds + loads ~1,163 CPT codes
│       └── load_icd10_codes.py    # Embeds + loads ~74,260 ICD-10 codes
├── data/
│   ├── all-2025-cpt-codes.csv     # Source CPT codes
│   └── icd10cm-codes-2025.txt     # Source ICD-10 codes (undotted format, e.g. E119)
├── evaluation/
│   ├── evaluate.py                # Precision@K / Recall@K / MRR / latency harness
│   ├── query_cli.py               # NEW: interactive keyword -> codes REPL
│   └── test_cases.json            # 10 hand-labeled test cases (dotted format, e.g. E11.9)
└── streamlit_app/                 # Chat UI (hits the FastAPI backend)
```

**Query flow:** keyword/description → `EmbeddingService.generate_embedding()`
(Voyage, `input_type="query"`) → parallel vector search (pgvector) + keyword
search (Postgres full-text) → Reciprocal Rank Fusion → top-K CPT/ICD-10 codes.

## What changed for Voyage-4-lite

| File | Change |
|---|---|
| `backend/requirements.txt` | added `voyageai` |
| `backend/app/config.py` | `VOYAGE_API_KEY`, `VOYAGE_MODEL_NAME`, `EMBEDDING_DIM=1024`; `extra="ignore"` fix for pydantic-settings |
| `backend/.env.example` | Voyage env vars instead of `EMBEDDING_MODEL_NAME` |
| `backend/app/services/embeddings.py` | rewritten to call `voyageai.Client`, with retry/backoff and `input_type` set per document vs. query |
| `backend/app/services/hybrid_search.py` | query embedding call wrapped in `asyncio.to_thread` (Voyage's client is a blocking network call) |
| `backend/scripts/setup_database.py` | dimension driven by `EMBEDDING_DIM`; `--reset` flag; vector index creation split out (see below) |
| `backend/scripts/load_cpt_codes.py` / `load_icd10_codes.py` | pass Voyage credentials explicitly; rebuild vector index after loading |
| `evaluation/evaluate.py` | `--file` flag; normalizes dotted vs. undotted ICD-10 codes before scoring |
| `evaluation/query_cli.py` | new interactive CLI |
| `backend/app/services/llm_service.py` | removed (was Perplexity-based reranker) |
| `backend/app/services/reranker_service.py` | new: `BAAI/bge-reranker-v2-m3` cross-encoder reranker for Expert mode |
| `backend/app/config.py` | removed `PERPLEXITY_API_KEY`/`PERPLEXITY_MODEL`; added `RERANKER_MODEL_NAME` |
| `backend/requirements.txt` | removed `openai`; kept `sentence-transformers` (now used for the reranker, not embeddings) |

### Two bugs fixed along the way
1. **ivfflat index built on an empty table** produces degenerate clusters and
   badly wrong nearest-neighbor results. Fixed by moving vector index
   creation to *after* data load (`rebuild_cpt_vector_index()` /
   `rebuild_icd10_vector_index()` in `setup_database.py`, called at the end
   of each loader script).
2. **ICD-10 code format mismatch**: source data stores codes undotted
   (`E119`), but `test_cases.json` uses clinical dotted notation (`E11.9`).
   `evaluate.py` now normalizes both before comparing.

## Reranker: Perplexity LLM → local BAAI/bge-reranker-v2-m3

Expert mode previously sent the top hybrid-search candidates to Perplexity
(Llama 3.1 Sonar) to rerank and explain them. That's replaced with a local
open-source cross-encoder ([BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3),
MIT licensed, ~568M params) via `sentence-transformers.CrossEncoder`:

- No API key, no network call, no per-query cost.
- Scores `(query, code description)` pairs directly (cross-encoder), then
  sigmoid-normalizes to 0-1 for `confidence_score`.
- No natural-language reasoning text — a cross-encoder only outputs a
  relevance score, not text — so `reasoning` is always `null` and
  `explanation` just names the reranker used.
- **Forces `device="cpu"`**: concurrent `predict()` calls on Apple's MPS
  (GPU) backend from multiple threads crashed the process during testing;
  CPU scoring runs sequentially in a single background thread instead.
- Only reorders the same candidate pool hybrid search already returned
  (top 10 per code type) — it can't add codes hybrid search missed, so
  Precision@5/Recall@5 are bounded by Quick mode's results. Its effect is
  on ranking quality within that pool (see MRR in results below).

## Steps to reproduce / run

### 1. Prerequisites
- A Voyage AI API key ([dashboard.voyageai.com](https://dashboard.voyageai.com/)) with a payment method on file (the free tier without one is capped at 3 requests/min, too slow for ~75k codes)
- A Postgres database with the `pgvector` extension (e.g. [Neon](https://neon.tech), paid tier — the free 512 MB tier is not enough for ~75k 1024-dim vectors)

### 2. Configure environment
```bash
cd backend
cp .env.example .env
# Edit .env: set NEON_DATABASE_URL, VOYAGE_API_KEY
# RERANKER_MODEL_NAME defaults to BAAI/bge-reranker-v2-m3, no API key needed
```

### 3. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Create the schema
```bash
python3 scripts/setup_database.py --reset   # --reset drops existing tables (needed when the embedding dimension changes)
```

### 5. Load and embed the code sets
```bash
python3 scripts/load_cpt_codes.py
python3 scripts/load_icd10_codes.py          # ~74k codes, takes a few minutes
# Optional: validate on a small sample first
python3 scripts/load_icd10_codes.py --limit 1000
```

### 6. Start the API
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 7. Spot-check with the interactive CLI
```bash
cd ../evaluation
python3 query_cli.py
# type keywords, e.g. "chest x-ray for pneumonia", "appendectomy"
python3 query_cli.py --mode expert   # exercises the local cross-encoder reranker
```

Or test Expert mode directly:
```bash
curl -s -X POST http://127.0.0.1:8000/api/code-suggestions \
  -H "Content-Type: application/json" \
  -d '{"clinical_description": "appendectomy", "search_mode": "expert", "max_results": 5}'
```
The first Expert-mode call downloads and loads `BAAI/bge-reranker-v2-m3`
(~2GB, one-time), so it's slower; subsequent calls reuse the loaded model.

### 8. Run the full evaluation
```bash
python3 evaluate.py                          # uses test_cases.json by default
python3 evaluate.py --file my_test_cases.json # or your own test cases
```

## Results (10 test cases, quick mode, voyage-4-lite)

| Metric | ICD-10 | CPT |
|---|---|---|
| Precision@5 | 20.0% | 10.0% |
| Recall@5 | 48.3% | 30.0% |
| MRR | 0.475 | 0.250 |
| Avg latency | ~230ms | — |

CPT precision/recall is lower because several test cases expect generic
E&M visit codes (99213/99214) for diagnosis-only queries — those codes
can't be inferred from CPT description text alone via embeddings.
Procedure-style queries (e.g. "appendectomy") score well in spot checks.

### Quick vs. Expert mode (reranker effect)

| Metric | ICD-10 Quick | ICD-10 Expert | CPT Quick | CPT Expert |
|---|---|---|---|---|
| Precision@5 | 20.0% | 20.0% | 10.0% | 10.0% |
| Recall@5 | 48.3% | 48.3% | 30.0% | 30.0% |
| MRR | 0.475 | **0.550** | 0.250 | 0.233 |
| Avg latency | ~230ms | ~423ms | — | — |

As expected, Precision@5/Recall@5 are identical — the reranker only
reorders the same candidate pool Quick mode already retrieves, it can't
add codes that weren't found. ICD-10 MRR improved (0.475 → 0.550): the
cross-encoder pushes more correct answers closer to rank 1. CPT MRR was
roughly flat (0.250 → 0.233) on this small 10-case set. Latency roughly
doubles due to the cross-encoder inference pass.

```bash
python3 evaluate.py   # answer 'y' to "Run evaluation in EXPERT mode?" to reproduce this
```

### Procedure-only breakdown

7 of the 10 test cases expect generic E&M visit codes (99213/99214/...)
for diagnosis-only queries with no procedure mentioned — no embedding
model can infer a visit-complexity code from diagnosis text alone. Isolating
just the 3 genuinely procedure-based test cases (`evaluation/test_cases_procedures.json`:
knee arthroplasty, aortic aneurysm repair, finger amputation) gives a much
clearer read on retrieval quality:

| Metric | ICD-10 | CPT |
|---|---|---|
| Precision@5 | 6.7% | 33.3% |
| Recall@5 | 33.3% | **100.0%** |
| MRR | 0.083 | 0.833 |

CPT Recall@5 hits 100% — the correct procedure code is in the top 5 every
time. ICD-10 recall drops on these same queries, because a query like
"total knee arthroplasty" describes the *procedure*, not the diagnosis
(`M17.9`, osteoarthritis) — there's little diagnosis-related vocabulary
for the embedding to match against.

**Takeaway**: voyage-4-lite retrieval is strong when the query's vocabulary
matches the code type's description vocabulary — procedure-phrased queries
favor CPT, diagnosis-phrased queries favor ICD-10. Cross-type queries
underperform because of a genuine semantic gap between a procedure
description and its associated diagnosis code (or vice versa), not because
of an embedding-quality defect.

```bash
python3 evaluate.py --file test_cases_procedures.json
```
