# Subconscious

Self-correcting cognitive memory layer for local LLM agents. Built for the Midas agent in [orion-ane](https://github.com/MidasMulli/orion-ane), but the storage backend, retrieval engine, and maintenance loops are general-purpose. Not Apple-Silicon-specific — ANE acceleration is optional, CPU fallback built in.

The user never sees the Subconscious. They notice the agent remembers what was shipped, what was killed, and what's still open across sessions.

---

## Architecture

```
                    ┌──────────────────────────┐
                    │   Multi-path retrieval   │  5-signal fusion:
                    │    (multi_path_retrieve) │   embedding + entity + type
                    └────────────┬─────────────┘   + impact + recency
                                 │
                    ┌────────────┴─────────────┐
                    │      LocalMemoryStore    │  SQLite WAL + numpy float32
                    │  (orion-ane/memory/...)  │  matrix. ~14 MB / 4,500 mems.
                    └────────────┬─────────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                                                  │
   ┌────▼─────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
   │  decay   │  │ consol  │  │ contra  │  │ vault   │    │
   │ (loop 1) │  │ (loop 2)│  │ (loop 3)│  │ (loop 4)│    │
   └──────────┘  └─────────┘  └─────────┘  └─────────┘    │
   ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
   │ prod     │  │ semant  │  │ canon   │  │  meta   │    │
   │ sync     │  │ supers  │  │ inject  │  │ inject  │    │
   │ (loop 5) │  │ (loop 6)│  │ (loop 7)│  │ (loop 8)│    │
   └──────────┘  └─────────┘  └─────────┘  └─────────┘    │
   ┌──────────┐                                            │
   │  vault   │                                            │
   │  sweep   │                                            │
   │ (loop 9) │                                            │
   └──────────┘                                            │
        │                                                  │
        └────────► all 9 loops orchestrated by ────────────┘
                  maintenance.run_all() (hourly via launchd)
```

## The 9 maintenance loops

1. **`decay_scores`** — exponential decay on `relevance_score` (7-day half-life). Below `MIN_RELEVANCE=0.05` a memory is dormant.
2. **`consolidate_duplicates`** — merge near-duplicate facts (cosine ≥ 0.85).
3. **`resolve_contradictions`** — idle-time contradiction resolution via the 70B verifier on candidate pairs.
4. **`vault_sync`** — supersede memories that conflict with canonical knowledge files (`vault/knowledge/*.md`).
5. **`production_state_sync`** — keep the live production state mirrored into memory so the agent always knows what's running.
6. **`semantic_supersession`** — three-signal supersession that catches paraphrased stale-state memories: `cosine similarity` + `tense classifier` + `restate-vs-contradict detector`. Catches what exact-string supersession misses.
7. **`canonical_state_inject`** — parse `CLAUDE.md` tables (production critical path, services, active projects, dead paths) into first-class canonical-state memories with stable IDs (`canonical_<section>_<slug>`). Idempotent. Includes a contradiction scan with 4 signals (similarity 0.50 + entity binding + non-canonical + non-past-finding) so updating a number in `CLAUDE.md` automatically supersedes the conflicting present-tense noise.
8. **`meta_memory_inject`** — parse session-log bullets into first-class activity memories with `source_role=meta`. Lets the agent answer "what did we ship today" / "what changed in the last build" from real history rather than hallucinating. Auto-orphans entries that age out of the 200-bullet active window.
9. **`vault_sweep`** — scan `agent_reports/`, `ane-reverse/`, `~/models/`, sibling code repos for completed deliverables that no knowledge file references. **Surfaces unwired completed work.** Closes the recurring failure mode where five sessions in a row had a research agent stop on prior art that was sitting on disk but never indexed.

## Multi-path 5-signal retrieval

`multi_path_retrieve.py` does fusion ranking over a candidate pool from cosine top-K:

| Signal | Weight | What it scores |
|---|---|---|
| **embedding** | 0.35 | cosine similarity over MiniLM-L6-v2 |
| **entity** | 0.25 | query entities ∩ memory `atom_entities` overlap |
| **type** | 0.15 | query category matches `atom_type` |
| **impact** | 0.15 | query topic in memory `atom_impacts` |
| **recency** | 0.10 | exponential decay on timestamp |

Plus:

- **Canonical boost** (`1.30×`) — memories with `source_role=canonical` outrank noise on the same topic.
- **Activity-query category override** — queries like "what did we ship today" / "tell me about Build X" / "catch me up" route to `project_status` category, swap canonical boost for meta boost, widen candidate pool 4×.
- **Filters** — drop user/assistant chat-turn echoes, drop raw vault file dumps, drop dedicated atoms with `source_role` in `{user, assistant}`.
- **Presentation layer** — `present()` formats top-K memories for the synthesizer with category-aware headers (`RELEVANT MEASUREMENTS:`, `RELEVANT DECISIONS:`, `RELEVANT CONNECTIONS:`).

## Structured atoms

Every memory has:

```python
{
  "text": "Llama 3.1-8B Q8 runs at 7.9 tok/s with 72 CoreML dispatches.",
  "type": "state",
  "source_role": "canonical",
  "timestamp": "2026-04-07T15:30:00",
  "atom_type": "quantitative",
  "atom_entities": ["llama-3.1-8b", "ane", "tok/s"],
  "atom_impacts": ["baseline_throughput", "production_stack"],
  "atom_tense": "present",
  "atom_confidence": 1.0,
  "atom_core": "Llama 3.1-8B Q8 ANE 7.9 tok/s",
  "atom_schema_version": 1
}
```

Retrieval uses every field. Storage uses `SQLite WAL` for ACID atomicity and an in-memory `numpy float32` matrix for sub-millisecond cosine via single matmul.

## CoreML MiniLM embedder

`_embedder.get_embedder()` is a factory that prefers a precompiled CoreML MiniLM-L6-v2 routed through ANE via `CPU_AND_NE` compute unit:

| Path | Latency | Throughput | Cosine vs CPU |
|---|---|---|---|
| **CoreML CPU_AND_NE (ANE)** | **0.84 ms/embed** | **1,197 embeds/s** | **0.999985** |
| CPU SentenceTransformer | 2.68 ms/embed | 373 embeds/s | baseline |

Falls back to CPU SentenceTransformer if the artifact is missing or `MIDAS_DISABLE_COREML_EMBED=1`.

## LocalMemoryStore

The storage backend lives in [orion-ane/memory/local_store.py](https://github.com/MidasMulli/orion-ane/tree/main/memory). It replaced ChromaDB in Main 24 after a Track 2 banking purge wedged the ChromaDB rust binding's `get_collection` indefinitely. Building our own gave us:

- SQLite WAL mode (set by us, controlled busy_timeout)
- One ACID-atomic write path
- In-memory numpy float32 matrix → sub-ms cosine via single matmul (no HNSW, no rust deadlock)
- Drop-in `_CollectionShim` that exposes the chromadb `collection.{get,query,upsert,update,delete}` API surface so existing call sites work unchanged
- 13.9 MB on disk for ~4,500 memories
- ~30-second migration from any pre-existing snapshot

## Files

| File | Role |
|---|---|
| `multi_path_retrieve.py` | 5-signal fusion + presentation layer |
| `canonical_inject.py` | loop 7 — parse CLAUDE.md → canonical memories |
| `meta_memory_inject.py` | loop 8 — parse session log → meta memories |
| `semantic_supersede.py` | loop 6 — three-signal paraphrase supersession |
| `vault_sweep.py` | loop 9 — surface unwired deliverables |
| `maintenance.py` | orchestrator for all 9 loops + launchd entry point |
| `extractor/claim_splitter.py` | CPU FactExtractor — atomic claim splitting |
| `extractor/rule_classifier.py` | CPU FactExtractor — type + domain regex classifier |
| `extractor/assembler.py` | CPU FactExtractor — assemble typed claims into memory atoms |
| `_embedder.py` | shared embedder factory (CoreML preferred, CPU fallback) |
| `conceptual_extractor.py` | secondary extractor for non-atomic conceptual chunks |
| `entity_enricher.py` | populate per-entity wikilink notes from memory atoms |
| `retrieval_logger.py` | gold-set evaluation harness (Hit@5, fact recall, contamination) |

## Measurements

- **Hit@5: 100% (20/20)** on the gold set after canonical-state injection (Main 22)
- **Cross-session continuity: 100%** across 5 sessions (6/6 references resolved, 24/24 turns coherent)
- **Vault sweep first run: 95% unreferenced** (428 of 451 scanned) — surfaced multiple completed deliverables that had been built but never wired into production
- **System recall: 83%** combined (8B ANE extractor + CPU FactExtractor) on the gold set, vs 76% solo and 65% CPU-only

## Related

- [orion-ane](https://github.com/MidasMulli/orion-ane) — the Midas agent and `LocalMemoryStore` backend
- [ane-compiler](https://github.com/MidasMulli/ane-compiler) — the 8B Q8 extractor that feeds the Subconscious
- [ngram-engine](https://github.com/MidasMulli/ngram-engine) — the verifier server with prefix-cache integration

## License

MIT.
