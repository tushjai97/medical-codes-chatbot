# Project Writeup: Swapping to Voyage-4-lite Embeddings + Local Reranker

This is a narrative account of the work done on the `voyage-4-lite-embeddings`
branch: what the goal was, what was actually done, and — in detail — every
issue hit along the way and how it was diagnosed and fixed. For a quick
reference (file-by-file changes, setup commands, results tables) see
[VOYAGE_INTEGRATION.md](VOYAGE_INTEGRATION.md). This document is the longer
story behind those changes.

## Goal

The `medical-codes-chatbot` repo is a RAG pipeline that maps clinical
keyword queries to CPT/ICD-10 billing codes, using Postgres + pgvector for
hybrid (vector + keyword) search. It originally used a local, free embedding
model (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) and Perplexity's
LLM API for optional reranking in "Expert mode."

The task: swap the embedding model to Voyage AI's `voyage-4-lite`
(a hosted, paid embedding API) and measure retrieval performance —
Precision@K, Recall@K, MRR, and latency — using a simple interactive
keyword-in/codes-out CLI, plus the repo's existing evaluation harness.
Later in the session, the goal expanded to also replace the Perplexity
LLM reranker with an open-source, locally-run alternative.

## Part 1: Swapping the embedding model

### What changed

`EmbeddingService` (`backend/app/services/embeddings.py`) was rewritten
from a `SentenceTransformer.encode()` wrapper to a `voyageai.Client`
wrapper. Two details mattered for correctness, not just plumbing:

- **`input_type` asymmetry**: Voyage's models are trained to benefit from
  knowing whether text being embedded is a search *query* or a *document*.
  Code descriptions (embedded once, at load time) use `input_type="document"`;
  the user's search text (embedded per request) uses `input_type="query"`.
  Getting this backwards doesn't error — it just quietly degrades retrieval
  quality, so it was easy to get right structurally (two different call
  sites) rather than relying on remembering to pass the right value.
- **Dimension change**: `voyage-4-lite` defaults to 1024-dim output
  (confirmed against Voyage's docs, with Matryoshka options for 256/512/2048
  also available). The old 384-dim `sentence-transformers` model and the new
  1024-dim Voyage model are not interchangeable — every `vector(384)` column
  in Postgres had to become `vector(1024)`, and every embedding in the
  database had to be regenerated (embeddings from different models aren't
  comparable to each other).

Since Voyage's client makes a blocking network call, and the query-embedding
call site in `hybrid_search.py` is inside an `async def`, that call is now
wrapped in `asyncio.to_thread(...)` so a slow Voyage API call doesn't stall
the FastAPI event loop for other concurrent requests.

### Issue 1: `pydantic-settings` rejected the existing `.env` file

**Symptom**: `pydantic_core._pydantic_core.ValidationError: 3 validation
errors for Settings ... Extra inputs are not permitted [type=extra_forbidden]`
for `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`.

**Cause**: pre-existing, unrelated to the Voyage swap. The version of
`pydantic-settings` that got installed defaults to rejecting any `.env`
variable that isn't declared as a field on the `Settings` class — and the
repo's own `.env.example` includes optional LangSmith tracing variables that
were never declared in `config.py`.

**Fix**: added `extra = "ignore"` to `Settings.Config`, so undeclared env
vars are silently ignored instead of raising. One-line fix, but it blocked
every script from starting until diagnosed.

### Issue 2: Voyage API rate-limited to 3 requests/minute

**Symptom**: `voyageai.error.RateLimitError: You have not yet added your
payment method... reduced rate limits of 3 RPM and 10K TPM.`

**Cause**: a new Voyage account without a payment method on file is capped
hard, regardless of the free token quota. At 3 requests/minute, embedding
~75,000 ICD-10 descriptions in batches would have taken many hours.

**Fix**: this required the user to add a payment method in the Voyage
dashboard (an action I can't take on someone's behalf — entering billing
details is out of scope for what I do automatically). Once added, the
rate limit lifted within minutes and the full load completed normally.

### Issue 3: Neon's free-tier storage limit (512 MB) was too small

**Symptom**: partway through inserting ~74k ICD-10 rows,
`asyncpg.exceptions.DiskFullError: could not extend file because project
size limit (512 MB) has been exceeded`. Only 34,000 of 74,260 rows made it
in before the insert failed.

**Cause**: 1024-dim float vectors are ~2.7x the storage of the old 384-dim
vectors, and 74k rows of code descriptions + `tsvector` full-text index +
`ivfflat` vector index columns add up. Direct measurement
(`pg_database_size`) showed 491 MB used for a partial load, confirming this
wasn't a fluke — the full dataset genuinely doesn't fit in 512 MB at this
dimension.

**Resolution**: the user upgraded their Neon plan. After that, dropping and
recreating both tables (`setup_database.py --reset`) and reloading gave a
clean 1028 MB database with all 1,163 CPT + 74,260 ICD-10 rows and their
1024-dim embeddings.

**Note on a dashboard discrepancy**: at one point the user's Neon dashboard
showed "35 MB" storage used, which contradicted Postgres's own
`pg_database_size()` report of 491 MB (and the hard error Neon itself had
just thrown at that size). This was never fully root-caused — it's most
likely a lag in Neon's UI storage metric or a different metric entirely
(e.g. "logical size" vs. physical/WAL-inclusive size) — but the two direct,
reproducible signals (the actual Postgres size function and the enforced
error) were trusted over the dashboard number, since the error itself
proved the physical limit was real.

### Issue 4: `ivfflat` vector index built on an empty table (the big one)

This was the most consequential bug of the whole task, and it produced the
most misleading symptom: after the first full data load, a test query for
**"chest x-ray for pneumonia"** returned completely unrelated ICD-10 codes
— a maxilla fracture, dental pulp diseases — with a top confidence score of
1.0. Nothing about "pneumonia" or "chest" or "x-ray" was anywhere near the
top of the results.

**Diagnosis process**: the confidence score being 1.0 for a nonsense result
was the first clue this wasn't just "the embedding model isn't very good" —
a 1.0 confidence is just min-max normalization within the returned result
set (see `ranking.py::normalize_scores`), so it says nothing about absolute
quality; the *ranking itself* was the problem, not the score display. That
ruled out a display bug and pointed at retrieval.

The embedding model itself was verified independently and worked correctly
in isolation (a direct Voyage API call for "chest x-ray" produced a sane
1024-dim vector). The pgvector cosine query
(`vector_search.py`, using the `<=>` operator) was also structurally
correct — cosine distance via `<=>` doesn't require pre-normalized vectors,
so that wasn't it either.

The actual cause: `setup_database.py`'s original flow created the
`ivfflat` indexes (`CREATE INDEX ... USING ivfflat ... WITH (lists = 100)`)
**before** any data was loaded into the tables — schema setup ran first,
then the loader scripts populated the tables afterward. An `ivfflat` index
works by clustering the *existing* vectors into `lists` buckets via k-means
at `CREATE INDEX` time, then only searching the nearest few buckets at query
time (approximate, not exact, nearest-neighbor search). Building that index
against an **empty** table means there's no data to cluster — the index
ends up with degenerate, essentially meaningless bucket boundaries. Every
row inserted afterward gets assigned to a bucket based on that broken
clustering, and searches only look inside 1-few buckets by default, so the
genuinely-nearest neighbors can easily be sitting in a bucket the search
never looks at.

**Fix**: rebuilt the two vector indexes (`REINDEX INDEX idx_cpt_vector;
REINDEX INDEX idx_icd10_vector;`) after the data was already loaded — this
alone fixed the pneumonia query immediately (pneumonia-related ICD-10 codes
correctly took the top ranks). To make sure this class of bug couldn't
recur, `setup_database.py` was restructured: non-vector indexes (full-text
GIN, category/chapter B-tree) still get created upfront since order doesn't
matter for those, but `ivfflat` index creation was pulled out into
`rebuild_cpt_vector_index()` / `rebuild_icd10_vector_index()` helper
functions, and both loader scripts (`load_cpt_codes.py`,
`load_icd10_codes.py`) now call the relevant one automatically as their
final step, after all rows are inserted.

### Issue 5: ICD-10 code format mismatch (dotted vs. undotted)

This is the one the user specifically asked about, so here's the full
explanation.

**Background — two valid representations of an ICD-10-CM code**: ICD-10-CM
codes are conventionally written with a decimal point after the third
character once a code has more than 3 characters — e.g. `E11.9` (Type 2
diabetes mellitus without complications). That's the format used in clinical
documentation, EHRs, and billing paperwork. But the *raw source files* CMS/NCHS
publish for the annual code set — which is exactly what
`data/icd10cm-codes-2025.txt` in this repo is — store codes **without** the
decimal: `E119`. Both strings refer to the exact same code; the decimal is a
formatting/display convention, not part of the code's actual identity.

**Where this broke things**: the pipeline's own loader
(`load_icd10_codes.py`) parses that raw file as-is, so every `icd10_code`
value stored in Postgres is undotted (`E119`, `Z0000`, `M179`, ...). But
`evaluation/test_cases.json` — the hand-written ground-truth test cases —
was authored using clinical dotted notation (`E11.9`, `Z00.00`, `M17.9`,
...), because that's how a person naturally writes an ICD-10 code.

`evaluate.py`'s scoring functions (`calculate_precision_at_k`,
`calculate_recall_at_k`, `calculate_mrr`) compared predicted vs. expected
codes with a plain substring check:
`any(exp in code or code in exp for exp in expected)`. That check requires
one string to literally contain the other. `"E11.9"` is not a substring of
`"E119"` (there's a `.` character in one that's simply absent from the
other), and `"E119"` is not a substring of `"E11.9"` either. So even when
the API returned the *exact correct code*, the scoring function recorded it
as a miss — silently, with no error, just an artificially low score.

**How this was caught**: while reviewing the "SAMPLE RESULTS" printout
after a quick-mode eval run, the first test case's `predicted` list showed
`['E119', 'E139', 'E089']` scored as 0% precision against `expected:
['E11.9', 'E11']` — but `E119` is visibly the same code as `E11.9`,
just without the dot. That's what triggered the investigation into the
comparison logic rather than the retrieval logic.

**Fix**: added a `_normalize_code()` helper in `evaluate.py` that strips `.`
(and upper-cases) before comparing, applied consistently across all three
metric functions. Re-running the exact same evaluation, with the exact same
API, same database, same embeddings — nothing about retrieval changed —
ICD-10 Recall@5 jumped from **10.0% to 48.3%** and MRR from **0.133 to
0.475**. That jump is the size of the bug: roughly 4 in 10 correct
retrievals were being scored as wrong purely because of code formatting.

This is a good example of why isolating "is the model bad" from "is the
measurement bad" matters — the embedding model had been working reasonably
well the whole time; the harness measuring it was wrong.

## Part 2: Replacing the LLM reranker

### What changed and why

The repo's "Expert mode" sent the top hybrid-search candidates to
Perplexity's API (Llama 3.1 Sonar) for reranking and natural-language
reasoning. The user asked to replace this with an open-source alternative,
researched via web search for current (2026) options.

After comparing candidates — BGE-reranker-v2-m3, Qwen3-Reranker,
gte-reranker-modernbert-base, ms-marco-MiniLM-L-6-v2 — **BAAI/bge-reranker-v2-m3**
was chosen: MIT licensed, ~568M parameters, a well-established cross-encoder
that runs on CPU, and a good fit for short-text domain reranking (clinical
query vs. code description) at the scale this pipeline needs (reranking
~10-20 candidates per query, not a huge batch).

`llm_service.py` (the Perplexity wrapper) was deleted entirely and replaced
with `reranker_service.py`, using `sentence_transformers.CrossEncoder`.
Since a cross-encoder outputs a bare relevance score rather than generated
text, the `reasoning` field per-code is now always `null`, and the overall
`explanation` field just states which reranker was used rather than
containing LLM-generated clinical reasoning — a real, acknowledged loss of
functionality traded for removing the API dependency and cost entirely.

### Issue 6: concurrent cross-encoder calls crashed the server on Apple Silicon

**Symptom**: the very first live test of Expert mode returned no response
at all — `curl` got an empty body, and a retry a few seconds later failed to
connect entirely. The `uvicorn` process had died.

**Diagnosis**: the server log showed the reranker model had loaded
successfully (`Reranker model loaded`), logged `No device provided, using
mps` (Apple's Metal GPU backend), started two `Batches:` progress bars —
one for CPT candidates, one for ICD-10 candidates, since the initial
implementation scored both in parallel via `asyncio.gather` +
`asyncio.to_thread` — and then the process silently died with a leaked
semaphore warning right after both batches started.

**Cause**: the two parallel `to_thread` calls both invoked
`CrossEncoder.predict()` on the *same model instance* at the same time.
PyTorch's MPS backend isn't safe for concurrent use across threads the way
CPU inference generally is; running two overlapping forward passes on the
GPU backend from different threads crashed the process outright.

**Fix**: two changes. First, `CrossEncoder(model_name, device="cpu")`
explicitly forces CPU inference, sidestepping MPS thread-safety entirely.
Second, the CPT and ICD-10 scoring calls were changed from concurrent
(`asyncio.gather` over two `to_thread` calls) to sequential (both run
inside one `to_thread` call, one after the other) — on CPU there's no
speed benefit to "parallelizing" two calls into the same single-threaded
model instance anyway, so sequential execution is both safer and no slower
in practice. After this fix, repeated Expert-mode requests ran reliably
with no crashes.

## Results summary

Quick mode (hybrid search only, no reranking), 10 test cases:

| Metric | ICD-10 | CPT |
|---|---|---|
| Precision@5 | 20.0% | 10.0% |
| Recall@5 | 48.3% | 30.0% |
| MRR | 0.475 | 0.250 |

CPT numbers are pulled down by test cases that expect generic E&M visit
codes (99213/99214) for diagnosis-only queries with no procedure
mentioned — no embedding model can infer a visit-complexity billing code
from diagnosis text alone. Isolating just the 3 genuinely procedure-based
test cases shows CPT Recall@5 at 100%; see the "Procedure-only breakdown"
in `VOYAGE_INTEGRATION.md` for the full comparison.

Expert mode (same candidates, reranked by BAAI/bge-reranker-v2-m3):
Precision@5/Recall@5 are identical to Quick mode (the reranker only
reorders the same candidate pool, it can't retrieve codes hybrid search
missed), but ICD-10 MRR improved from 0.475 to 0.550 — correct answers
land closer to rank 1 — at roughly double the latency (~230ms → ~423ms
average) from the added cross-encoder inference pass.

## What this session illustrates

Almost every issue here was either an infrastructure/quota limit (Voyage
rate limits, Neon storage, a pydantic-settings version mismatch) or a
measurement bug (the dotted/undotted code mismatch) rather than a genuine
model-quality problem. The one real architectural bug — the `ivfflat`
index built before data existed — produced results bad enough to look like
a fundamentally broken embedding model, and the fastest way to tell those
apart was checking the pieces in isolation (embedding vector shape, raw SQL
query correctness) before concluding the model itself was at fault.
