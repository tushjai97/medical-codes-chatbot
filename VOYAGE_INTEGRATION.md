# Voyage-4-lite Embedding Integration

This branch (`voyage-4-lite-embeddings`) swaps the pipeline's embedding
model from local `sentence-transformers/all-MiniLM-L6-v2` (384-dim) to
Voyage AI's hosted `voyage-4-lite` (1024-dim), and adds tooling to spot-check
and measure retrieval performance.

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
│   │       └── llm_service.py     # Perplexity LLM reranking (Expert mode only)
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

### Two bugs fixed along the way
1. **ivfflat index built on an empty table** produces degenerate clusters and
   badly wrong nearest-neighbor results. Fixed by moving vector index
   creation to *after* data load (`rebuild_cpt_vector_index()` /
   `rebuild_icd10_vector_index()` in `setup_database.py`, called at the end
   of each loader script).
2. **ICD-10 code format mismatch**: source data stores codes undotted
   (`E119`), but `test_cases.json` uses clinical dotted notation (`E11.9`).
   `evaluate.py` now normalizes both before comparing.

## Steps to reproduce / run

### 1. Prerequisites
- A Voyage AI API key ([dashboard.voyageai.com](https://dashboard.voyageai.com/)) with a payment method on file (the free tier without one is capped at 3 requests/min, too slow for ~75k codes)
- A Postgres database with the `pgvector` extension (e.g. [Neon](https://neon.tech), paid tier — the free 512 MB tier is not enough for ~75k 1024-dim vectors)

### 2. Configure environment
```bash
cd backend
cp .env.example .env
# Edit .env: set NEON_DATABASE_URL, VOYAGE_API_KEY, PERPLEXITY_API_KEY (dummy value ok if not using Expert mode)
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
```

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
