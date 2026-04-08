"""Main 23 Build 2: Multi-path retrieval with 5-signal fusion + presentation layer.

Replaces single-cosine recall with a ranked fusion of:
  1. Embedding similarity (0.30)  — cosine on the document text (existing path)
  2. Entity match           (0.25)  — query entities ∩ memory atom_entities
  3. Type match             (0.20)  — query category matches memory atom_type
  4. Impact chain           (0.15)  — query topic in memory atom_impacts
  5. Temporal recency       (0.10)  — exponential decay on timestamp

Plus the existing canonical-state boost (1.30x for source_role=canonical) from
Main 22 Build 3 / Main 23 Build 0, which is multiplied at the end.

The fusion fires on memories that have the `atom_*` schema fields written by
Agent 2's heuristic migration. Memories without atoms fall back to pure
cosine similarity (no penalty, no boost from the structured signals).

Presentation layer: when injecting top-k memories into the synthesizer
prompt, format them per query category instead of dumping raw text.
"""
from __future__ import annotations
import re
import json
import time
import numpy as np

# ---------------------------------------------------------------------------
# Fusion weights (tunable; starting points per directive)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "embedding": 0.35,   # bumped from 0.30 — Build 2 validation showed type was
    "entity":    0.25,   # flattening cosine differences across all state-type
    "type":      0.15,   # memories, costing the contamination regression on
    "impact":    0.15,   # "active projects" query.
    "recency":   0.10,
}
CANONICAL_BOOST = 1.30  # mirrors daemon.py MemoryStore.recall
META_BOOST = 1.15       # baseline: meta beats nothing in particular
META_BOOST_ACTIVITY = 1.55  # activity-query path: meta dominates canonical
CANONICAL_BOOST_ACTIVITY = 1.0  # canonical loses its boost on activity queries

# Activity-query detector. When the user asks about session work (what we
# shipped, what changed, tell me about Build X, today, catch me up), we want
# session_activity meta memories to outrank canonical-state memories. The
# default weighting overshoots toward canonical because canonicals are
# query-syntactic (cosine 0.20-0.30) and sit above the meta-bullet floor.
import re as _re_act
_ACTIVITY_RE = _re_act.compile(
    r"\b(?:"
    r"ship(?:ped)?|change[ds]?|build|built|complete[ds]?|fix(?:ed)?|"
    r"today|recent(?:ly)?|catch me up|"
    r"what (?:did|have) (?:we|i)|tell me about (?:main|build)|"
    r"what (?:changed|happened|shipped)|"
    r"what'?s new"
    r")\b",
    _re_act.IGNORECASE,
)


def is_activity_query(query: str) -> bool:
    return bool(_ACTIVITY_RE.search(query))


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------
QUERY_CATEGORY_PATTERNS = {
    "project_status": [
        r"\bwhat'?s active\b", r"\bactive (?:projects?|right now)\b",
        r"\bwhat'?s parked\b", r"\bpriorities?\b",
        r"\bwhat (?:should|am) I work on\b", r"\bstatus of\b",
        r"\bwhat'?s (?:happening|going on)\b", r"\bcatch me up\b",
    ],
    "technical": [
        r"\bwhat (?:is|was|are) the\b", r"\bhow (?:fast|much|many)\b",
        r"\btok/s\b", r"\bms/tok\b", r"\bgb/s\b",
        r"\bdispatch\b", r"\blatency\b", r"\bthroughput\b",
        r"\bmeasured?\b", r"\bbenchmark\b",
    ],
    "cross_domain": [
        r"\baffect(?:s|ed)?\b", r"\brelationship\b", r"\bbetween\b.*\band\b",
        r"\bhow does .+ (?:affect|relate to|impact)\b",
        r"\bwhat .+ inform\b", r"\bif .+ change\b", r"\bif .+ upgrade\b",
    ],
    "adversarial": [
        r"\bshould (?:we|i) (?:revisit|try|look at)\b",
        r"\b(?:eagle|living model|cache.swap|drafter on gpu)\b",
        r"\bwhat about\b", r"\b(?:can|could) we\b",
    ],
}


def classify_query(query: str) -> str:
    ql = query.lower()
    scores = {cat: sum(1 for p in pats if re.search(p, ql))
              for cat, pats in QUERY_CATEGORY_PATTERNS.items()}
    if not any(scores.values()):
        return "technical"
    return max(scores, key=scores.get)


KNOWN_ENTITIES = {
    "qwen2.5-72b", "qwen-72b", "qwen 72b", "qwen3.5-0.8b", "qwen-0.8b",
    "llama-3.1-8b", "llama-8b", "llama 8b", "llama-1b", "llama 1b",
    "llama-3.3-70b", "llama-70b", "llama 70b", "neuron", "gpt-2",
    "eagle-3", "eagle",
    "ane", "gpu", "cpu", "amx", "metal", "nax", "slc", "dram", "sram",
    "m5 pro", "m5", "m4",
    "subconscious", "spec decode", "ane-compiler", "ane-dispatch", "ane-toolkit",
    "midas", "vault", "knowledge", "paper", "living model", "chimera",
    "q8", "q4", "q3", "fp16", "bf16", "tok/s", "ms/tok", "gb/s",
    "production", "verifier", "drafter", "extraction", "retrieval",
    "memory", "throughput", "baseline",
}

IMPACT_TOPICS = {
    "spec_decode": ["spec decode", "speculative", "drafter", "n-gram", "verifier"],
    "subconscious": ["subconscious", "memory", "recall", "extraction"],
    "ane": ["ane", "neural engine", "dispatch", "fusion"],
    "production_stack": ["production", "infrastructure", "service", "port"],
    "baseline_throughput": ["tok/s", "throughput", "baseline"],
    "paper": ["paper", "publication", "arxiv"],
    "compiler": ["compiler", "ane-compiler", ".hwx", "espresso"],
    "retrieval": ["retrieval", "recall", "hit@", "embedding"],
}


def extract_query_entities(query: str) -> list[str]:
    ql = query.lower()
    return [e for e in KNOWN_ENTITIES if e in ql]


def extract_query_topics(query: str) -> list[str]:
    ql = query.lower()
    return [topic for topic, kws in IMPACT_TOPICS.items() if any(kw in ql for kw in kws)]


CATEGORY_TYPE_PREFERENCE = {
    "project_status": {"state", "decision", "task", "preference",
                       "session_activity"},  # Main 24 Build 1: route activity Qs to meta memories
    "technical":      {"quantitative", "fact", "state", "conceptual"},
    "cross_domain":   {"relationship", "conceptual", "fact", "decision"},
    "adversarial":    {"decision", "fact", "session_activity"},
}


# ---------------------------------------------------------------------------
# Per-memory signal scorers
# ---------------------------------------------------------------------------
def _score_entity(query_entities, memory_meta):
    if not query_entities:
        return 0.0
    # Prefer atom_entities (Agent 2 migration), fall back to entities
    # (canonical_inject + legacy memories use the unprefixed field).
    raw = memory_meta.get("atom_entities") or memory_meta.get("entities", "[]")
    try:
        mem_ents = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return 0.0
    mem_ents_lower = [str(e).lower() for e in (mem_ents or [])]
    if not mem_ents_lower:
        return 0.0
    hits = sum(1 for qe in query_entities if any(qe in me or me in qe for me in mem_ents_lower))
    return min(1.0, hits / len(query_entities))


def _score_type(query_category, memory_meta):
    # Prefer atom_type, fall back to type
    atype = memory_meta.get("atom_type") or memory_meta.get("type", "")
    if not atype:
        return 0.0
    preferred = CATEGORY_TYPE_PREFERENCE.get(query_category, set())
    return 1.0 if atype in preferred else 0.0


def _score_impact(query_topics, memory_meta):
    if not query_topics:
        return 0.0
    raw = memory_meta.get("atom_impacts", "[]")
    try:
        mem_impacts = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return 0.0
    if not mem_impacts:
        return 0.0
    hits = sum(1 for qt in query_topics if qt in mem_impacts)
    return min(1.0, hits / len(query_topics))


def _score_recency(memory_meta, half_life_days=7.0):
    ts_str = memory_meta.get("timestamp", "")
    if not ts_str:
        return 0.3
    try:
        from datetime import datetime
        ts = datetime.fromisoformat(ts_str).timestamp()
    except Exception:
        return 0.3
    age_days = (time.time() - ts) / 86400
    return 2 ** (-age_days / half_life_days)


# ---------------------------------------------------------------------------
# Multi-path recall — wraps daemon.MemoryStore
# ---------------------------------------------------------------------------
def multi_path_recall(query, store, n_results=5, candidate_pool=30, verbose=False):
    category = classify_query(query)
    q_entities = extract_query_entities(query)
    q_topics = extract_query_topics(query)
    activity = is_activity_query(query)
    # Activity queries are project-status-shaped by definition. Override
    # the category so the type-preference signal favors session_activity
    # memories (otherwise queries like "what did we ship today" fall through
    # to "technical" — which prefers canonical state — and meta loses).
    if activity:
        category = "project_status"

    # Main 24 Build 1: widen the cosine pool aggressively on activity queries.
    # Session-activity bullets sit at cosine 0.15-0.20, well below the
    # canonical-state floor of 0.30. The default pool of 100 misses them;
    # 400 reliably includes today's bullets. Cost is one matmul over 3,800
    # rows + 400 SQLite row fetches — sub-50 ms.
    effective_pool = candidate_pool * 4 if activity else candidate_pool
    cosine_results = store.recall(query, n_results=effective_pool)

    if not cosine_results:
        return []

    rescored = []
    for r in cosine_results:
        meta = r.get("metadata", {}) or {}
        # Main 24 Build 1: filter raw chat turns from the recall pool. They
        # have high cosine to lexically similar new queries (e.g. "Catch me
        # up on what we shipped recently") but they are query echos, not
        # facts — they crowd out canonical/meta memories on every project-
        # status query. The Subconscious extractor is the canonical path
        # for getting *facts* out of conversations.
        if meta.get("source_role") in ("user", "assistant"):
            continue
        # Main 24 Build 1: also filter raw vault file dumps (`[Foo.md] # Foo
        # — [2026-03-17] ...`). These are pre-extraction file content, not
        # facts, and they win on cosine for any query whose words happen to
        # appear in markdown notes (e.g. "ship" matched MacBook delivery
        # entries, drowning out actual session activity).
        text = r.get("text", "")
        if text.startswith("[") and ".md]" in text[:60]:
            continue
        cosine_sim = r.get("similarity", r.get("score", 0.0))

        s_emb = float(cosine_sim)
        s_ent = _score_entity(q_entities, meta)
        s_typ = _score_type(category, meta)
        s_imp = _score_impact(q_topics, meta)
        s_rec = _score_recency(meta)

        fused = (
            WEIGHTS["embedding"] * s_emb +
            WEIGHTS["entity"]    * s_ent +
            WEIGHTS["type"]      * s_typ +
            WEIGHTS["impact"]    * s_imp +
            WEIGHTS["recency"]   * s_rec
        )

        sr = meta.get("source_role")
        if sr == "canonical":
            fused *= CANONICAL_BOOST_ACTIVITY if activity else CANONICAL_BOOST
        elif sr == "meta":
            fused *= META_BOOST_ACTIVITY if activity else META_BOOST

        rescored.append({
            **r,
            "fused_score": fused,
            "signal_breakdown": {
                "embedding": round(s_emb, 3),
                "entity": round(s_ent, 3),
                "type": round(s_typ, 3),
                "impact": round(s_imp, 3),
                "recency": round(s_rec, 3),
            },
            "query_category": category,
        })

    rescored.sort(key=lambda r: r["fused_score"], reverse=True)
    if verbose:
        for i, r in enumerate(rescored[:n_results], 1):
            sb = r["signal_breakdown"]
            print(f"  {i}. fused={r['fused_score']:.3f}  "
                  f"e={sb['embedding']:.2f} ent={sb['entity']:.2f} "
                  f"typ={sb['type']:.2f} imp={sb['impact']:.2f} rec={sb['recency']:.2f}  "
                  f"| {r.get('text','')[:100]}")
    return rescored[:n_results]


# ---------------------------------------------------------------------------
# Presentation layer
# ---------------------------------------------------------------------------
def present(memories, query, max_chars=1200):
    """Format the top-N memories for synthesizer injection, tailored to the
    query category. Accepts both nested-metadata shape (from multi_path_recall
    directly) and flat shape (from MemoryBridge.recall after Main 24 Build 0
    wiring, which lifts source_role to the top level).
    """
    if not memories:
        return ""
    category = classify_query(query)
    headers = {
        "project_status": "RELEVANT PROJECT STATE:",
        "technical":      "RELEVANT MEASUREMENTS:",
        "cross_domain":   "RELEVANT CONNECTIONS:",
        "adversarial":    "RELEVANT DECISIONS:",
    }
    header = headers.get(category, "RELEVANT MEMORIES:")
    lines = [header]
    used = len(header)
    for m in memories:
        if isinstance(m, str):
            text = m.strip()
            sr = ""
        else:
            text = m.get("text", "").strip()
            # Try nested metadata first (multi_path_recall direct shape), then
            # flat top-level (MemoryBridge.recall shape after Main 24)
            meta = m.get("metadata", {}) or {}
            sr = meta.get("source_role") or m.get("source_role", "")
        if not text:
            continue
        prefix = "[canonical] " if sr == "canonical" else ""
        line = f"  - {prefix}{text}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/midas/Desktop/cowork/orion-ane/memory")
    from daemon import MemoryStore
    store = MemoryStore("/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_live")
    test_queries = [
        "What's active right now?",
        "What's the 8B tok/s on ANE?",
        "How does the model swap affect spec decode?",
        "Should we revisit Living Model?",
    ]
    for q in test_queries:
        print(f"\n=== {q} ===")
        results = multi_path_recall(q, store, n_results=5, verbose=True)
        print(f"  category={results[0]['query_category'] if results else '-'}")
