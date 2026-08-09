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
# M125.1 A — atomic canonical rows (A3 ingestion). Sentence-level factual
# records with empty entities/impacts fields, so the structured-signal
# portion of fused (entity+type+impact = 0.55 of the weight) always
# contributes 0 → fused ≈ 0.35*s_emb + 0.10*s_rec. For typical cosine
# 0.30-0.40 this lands at 0.20-0.25, below the 0.40 midas_ui threshold
# even with CANONICAL_BOOST 1.30 applied. Use a cosine-dominant scorer
# similar to claude_automemory, tuned so atomic rows outrank generic
# canonicals on factual queries without overwhelming curated auto-memory.
CANONICAL_ATOM_BOOST = 1.80
META_BOOST = 1.05       # M125.3 C Pattern 1a: retune from 1.15 offline eval +6.78pp M125.2F / +8.93pp M125C parent-synthesis top-20 surfacing
META_BOOST_ACTIVITY = 1.55  # activity-query path: meta dominates canonical
CANONICAL_BOOST_ACTIVITY = 1.0  # canonical loses its boost on activity queries
# Main 33 close: claude_automemory holds the curated finding files I write
# explicitly between sessions (~/.claude/projects/-Users-midas/memory/finding_*.md).
# These are the highest-confidence corrected memories — when they exist on
# a topic, they should outrank passively-extracted canonical CLAUDE.md state
# which may contain superseded claims. Set CLAUDE_AUTOMEMORY_BOOST > CANONICAL_BOOST.
CLAUDE_AUTOMEMORY_BOOST = 3.20
# claude_vault_realtime holds findings ingested by the realtime watcher from
# new vault writes. These are higher-quality than passive extraction but lower
# than curated auto-memory. Modest boost.
CLAUDE_VAULT_REALTIME_BOOST = 1.20
# M125.2 B — parent-synthesis class-specific boost (Pattern 2).
# Parent-synthesis markdown bodies (e.g. vault/agent_reports/m117_parent_synthesis.md)
# are ingested by realtime_enricher as source_role='claude_vault_realtime' with
# empty entities/type/impact atomic fields — so the structured-signal portion
# of fused scoring (entity+type+impact = 0.55 weight) always returns 0, and
# the 1.20× CLAUDE_VAULT_REALTIME_BOOST can't overcome short canonical rows
# whose 1.30× CANONICAL_BOOST rides on structured matches. Measured on T28
# ("Summarize the fix surface from M115 through M118"): M116/M117/M118
# parent syntheses land at fused rank 30-97, below the K=20 cross-encoder
# window. Fix parallels the claude_automemory cosine-dominant scorer —
# bypass the structured-signal floor so canonical-class long-body syntheses
# score on embedding match, which IS authoritative for these records.
# Class detection requires all three conditions (length + source_role +
# id/file match) to stay tight; Class A session-meta bullets (short, high-
# frequency) retain META_BOOST at 1.15 unchanged.
PARENT_SYNTHESIS_BOOST = 1.80
PARENT_SYNTHESIS_MIN_LEN = 800
# research_ingest holds vault-file dumps. These are unstructured pre-extraction
# content; useful as fallback but should not outrank actual findings.
RESEARCH_INGEST_PENALTY = 0.85


def _is_parent_synthesis(meta: dict, text: str) -> bool:
    """M125.2 B — Class-B detection predicate (Pattern 2).

    Parent-synthesis records are long-body realtime-ingested canonical content
    whose id/file carries the 'parent_synthesis' marker. Requires all three
    conditions to avoid false-positive boost of short meta bullets:
      1. text length > PARENT_SYNTHESIS_MIN_LEN (800 chars) — filters out
         short session-activity bullets and classifier outputs
      2. source_role == 'claude_vault_realtime' — filters out assistant/user
         echoes and canonical short rows
      3. 'parent_synthesis' substring in id or source/file path — filters out
         other vault documents (paper drafts, INDEX files, random findings)
    """
    if not meta:
        return False
    sr = meta.get("source_role") or ""
    if sr != "claude_vault_realtime":
        return False
    if len(text or "") <= PARENT_SYNTHESIS_MIN_LEN:
        return False
    id_str = str(meta.get("id") or "").lower()
    src_str = str(meta.get("source") or meta.get("file") or "").lower()
    return "parent_synthesis" in id_str or "parent_synthesis" in src_str

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


# M97 Fix 2 — per-query domain classifier. Mirrors canonical_inject.py's
# _DOMAIN_KEYWORDS taxonomy so inject-time and retrieval-time both see the
# same topic space.
_M97_DOMAIN_KEYWORDS = {
    "hardware_characterization": [
        "slc", "amcc", "macc", "dcs", "ane", "nax", "gpu", "amx",
        "bandwidth", "register", "kext", "ioreport", "tflops",
        "fabric", "silicon", "m5 pro", "m4", "dispatch", "mmio",
        "aned", "coreml", "hwx", "neuron",
    ],
    "midas_infrastructure": [
        "midas", "router", "delegation",
        "spec decode", "n-gram", "drafter", "verifier", "prompt cache",
    ],
    "subconscious_memory": [
        "subconscious", "memory", "extraction", "recall", "retrieval",
        "memorystore", "maintenance loop", "enricher", "supersession",
        "canonical", "vault", "chromadb", "minilm", "embedder",
    ],
    "ml_models_training": [
        "llama", "qwen", "gpt", "gemma", "tok/s", "fine-tune", "lora",
        "distill", "inference", "training",
    ],
    "ane_compiler": [
        "ane-compiler", "ane-dispatch", "fusion", "gelu", "espresso",
        "mlpackage", "opcode", "macroop",
    ],
    "paper_writing": [
        "paper", "draft", "abstract", "every cycle", "arxiv",
        "reviewer", "methodology", "citation", "thesis", "locomo",
    ],
}


def _m97_classify_query_topics(query: str) -> set:
    """Return the set of topic keys whose keywords appear in the query.
    Case-insensitive substring match; mirrors canonical_inject.py classifier."""
    if not query:
        return set()
    q = query.lower()
    matched = set()
    for topic, kws in _M97_DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                matched.add(topic)
                break
    return matched


# ---------------------------------------------------------------------------
# M122 Stream A4 — Narrow ranking boost (summary_over_specific_retrieval)
#
# Validated on 3 turns in M120 D (T52, T73, T86): the SPECIFIC canonical is
# in vault but summary-level records outrank it in top-K cosine. A4
# re-ranks within the pool when the query is specific-shape, promoting
# detail-containing records above summary-level records.
#
# Composes with Stream A1's canonical-reserve in present(): A4 lifts the
# detail record into top-K; if top-1, A1's reserve slot is unnecessary.
# Pilot attribution distinguishes which fix was load-bearing.
#
# Design discipline (directive §3.4 + §9):
#   - Under-fit beats over-fit. Classifier errs toward narrow triggering.
#   - K8 lever: magnitude tune 0.30 → 0.15 if over-correction surfaces.
#   - K9 lever: narrow patterns, prefer explicit numeric/path/quoted
#     markers over loose "what is X" phrasing.
#   - Only touches memory-recall top-K scoring; other retrieval paths
#     (tool calls, registry lookups, etc.) unaffected.
# ---------------------------------------------------------------------------
A4_BOOST_MAGNITUDE = 0.30  # K8 lever: 0.30 → 0.15 if over-correction

# Narrow specific-shape query patterns. Prefer explicit numeric/path/quoted
# markers over loose "what is X" phrasing (K9 discipline).
_A4_SPECIFIC_QUERY_PATTERNS = [
    # Numeric-shape: explicit metric asks
    r"\bwhat'?s?\s+the\s+\w+%",
    r"\bwhat\s+(?:is|was|are|were)?\s*(?:the\s+)?(?:throughput|percentage|rate|ratio|speed|latency|bandwidth|count|waste)\b",
    r"\bwhat\s+(?:is|was|are|were)?\s*(?:the\s+)?(?:tok/s|gb/s|ms|ns|percent)\b",
    r"\bhow\s+(?:many|much|fast|slow|big|long)\b",
    r"\bwhat\s+(?:was|is|were)\s+the\s+(?:measured|exact|specific|precise)\b",
    r"\bwhat\s+(?:was|is|were)\s+the\s+(?:number|value|score|result|cost|size|dimension|dims?)\b",
    r"\bwhat\s+(?:are|were)\s+the\s+(?:dimensions?|dims?|values?|scores?|numbers?|sizes?)\b",
    # Path-shape: file/directory asks
    r"\bwhat\s+file\b",
    r"\bwhich\s+file\b",
    r"\bwhat\s+directory\b",
    r"\bwhich\s+directory\b",
    r"\bwhere\s+(?:is|was)\s+the\s+file\b",
    # Explicit specificity modifiers
    r"\b(?:exact|exactly|specific|specifically|precise|precisely|measured)\b",
    # Waste/cost-per-unit asks (T73-shape)
    r"\b(?:waste|cost|overhead)\s+per\b",
    r"\bhow\s+much\s+\w+\s+(?:per|waste|cost|saved)\b",
    # Embed/dim asks (T52-shape)
    r"\bembed(?:ding)?\s+(?:dims?|dimensions?|size)\b",
    r"\binput\s+and\s+output\s+(?:embed|dims?|dimensions?)\b",
]
_A4_SPECIFIC_QUERY_RE = re.compile(
    "|".join(f"(?:{p})" for p in _A4_SPECIFIC_QUERY_PATTERNS),
    re.IGNORECASE,
)
# Quoted-string detector: query contains "..." or '...'.
_A4_QUOTED_RE = re.compile(r'"[^"]{2,}"' + r"|'[^']{2,}'")
# Numeric-anchor detector: query itself contains a percentage, unit, or
# ratio (e.g. "83%", "50 tok/s", "19/20"). When the user anchors the
# question with a concrete measurement, they are asking about that
# measurement's context — detail-shape records are the correct target.
_A4_QUERY_NUMERIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*%"
    r"|\b\d+(?:\.\d+)?\s*(?:tok/s|gb/s|ms|ns|tflops|gflops|x)\b"
    r"|\b\d+/\d+\b"
    r"|\btok/s\b|\bgb/s\b|\btflops\b|\bgflops\b",
    re.IGNORECASE,
)


def is_a4_specific_shape_query(query: str) -> bool:
    """Query-shape classifier for A4.

    Returns True when query asks for a measurable/specific value.
    Narrow by design: missing a few specific-shape queries is better
    than firing on summary-shape queries (K9 discipline).
    """
    if not query:
        return False
    if _A4_SPECIFIC_QUERY_RE.search(query):
        return True
    if _A4_QUOTED_RE.search(query):
        return True
    if _A4_QUERY_NUMERIC_RE.search(query):
        return True
    return False


# Detail-record patterns. A record is detail-shape if its text contains
# a decimal number, percentage, measurement unit, path-shape, quoted
# string, or big-number run (e.g. 4096, 5376).
_A4_DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")
_A4_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_A4_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:tok/s|tok/sec|gb/s|gb|tb|mb|kb|ms|ns|us|μs|w|watts?|ghz|mhz|khz|hz|tflops|gflops|mflops|x)\b",
    re.IGNORECASE,
)
_A4_PATH_RE = re.compile(
    r"(?:/\w+){2,}"
    r"|\b\w+\.(?:py|md|json|txt|c|h|m|swift|mm|mlpackage|mlmodelc|hwx|safetensors|yaml|yml|toml)\b"
)
_A4_QUOTED_BODY_RE = re.compile(r'"[^"]{3,}"' + r"|'[^']{3,}'")
# Ratio pattern (e.g. "19/20", "1/20", "5-10x").
_A4_RATIO_RE = re.compile(r"\b\d+/\d+\b|\b\d+-\d+x\b", re.IGNORECASE)
# Digit-sequence fallback: three-or-more-digit run (e.g. 4096, 5376).
_A4_BIGNUM_RE = re.compile(r"\b\d{3,}\b")


def is_a4_detail_record(text: str) -> bool:
    """Return True when text contains a detail-shape marker."""
    if not text:
        return False
    if _A4_PERCENT_RE.search(text):
        return True
    if _A4_DECIMAL_RE.search(text):
        return True
    if _A4_UNIT_RE.search(text):
        return True
    if _A4_PATH_RE.search(text):
        return True
    if _A4_QUOTED_BODY_RE.search(text):
        return True
    if _A4_RATIO_RE.search(text):
        return True
    if _A4_BIGNUM_RE.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# M125.1 Stream C — cross-encoder rerank.
#
# After fused-score sorting, re-score the top-K candidates with the
# ms-marco-MiniLM-L6-v2 cross-encoder and re-sort. Gated to specific-shape
# queries (reuses M122 A4 `is_a4_specific_shape_query`) so the ~22 ms CPU cost
# only fires on turns that need it (~5-10% of traffic).
#
# Addresses M125 A1 T28 residual: "Summarize the fix surface from M115
# through M118" — cosine/fusion place M116/M117/M118 parent syntheses at
# rank 11-15 against recency-boosted session meta-bullets. Cross-encoder
# with full query-document attention correctly lifts the parent syntheses.
#
# Design discipline (Pattern-1):
#   - Under-fit gating: classifier-narrow, not every query.
#   - Does NOT touch present() / canonical-reserve path (Stream A1).
#   - Does NOT touch prefix cache (operates on retrieval output only).
#   - Lazy load: no cost if rerank never fires in a process.
#   - Graceful fallback: any load / predict exception falls back to fused.
#
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, CPU).
# Measured latency: 21.7 ms median for K=20 pairs on M5 Pro CPU.
# ---------------------------------------------------------------------------
# M125.1 C ship-or-defer verdict: DEFER. Default disabled. Infrastructure
# lands clean (11/11 new tests + 36/36 regression tests PASS), latency
# measured (~107 ms CPU K=20, 35-81 ms MPS K=20-50), but T28 primary
# verification FAILS at K=20 because M116/M117/M118 parent syntheses sit
# at fused rank 30-46 — outside the rerank window. T28 full close requires
# either upstream pool-composition work (session-bullet dedup or role-weight
# retuning) or K=50 MPS (~81 ms, which introduces GPU contention with Gemma
# 4 31B Q4 verifier).
# Opt-in path (e.g. for A/B bench):
#   M125_C_RERANK_ENABLE=1   → turn rerank on
#   M125_C_RERANK_K=50       → wider window (closes T28 on CPU at 287 ms)
#   M125_C_RERANK_DEVICE=mps → MPS backend (closes T28 at 81 ms; GPU shared)
import os as _os_default
RERANK_ENABLED_DEFAULT = _os_default.environ.get("M125_C_RERANK_ENABLE") == "1"
RERANK_K_DEFAULT = int(_os_default.environ.get("M125_C_RERANK_K", "20"))
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANK_MODEL = None           # lazy singleton; None = not yet attempted
_RERANK_LOAD_FAILED = False    # sticky flag after a failed load

def _get_rerank_model():
    """Lazy load the cross-encoder. Returns model or None on failure."""
    global _RERANK_MODEL, _RERANK_LOAD_FAILED
    if _RERANK_MODEL is not None:
        return _RERANK_MODEL
    if _RERANK_LOAD_FAILED:
        return None
    import os as _os
    if _os.environ.get("M125_C_RERANK_DISABLE") == "1":
        _RERANK_LOAD_FAILED = True
        return None
    try:
        _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder  # type: ignore
        # CPU default: 21-22 ms for K=20, no Gemma-4 verifier GPU contention.
        # Opt-in MPS: ~3 ms/K pair but shares GPU with the verifier.
        _device = _os.environ.get("M125_C_RERANK_DEVICE", "cpu").strip().lower()
        if _device not in ("cpu", "mps"):
            _device = "cpu"
        _RERANK_MODEL = CrossEncoder(RERANK_MODEL_NAME, device=_device)
        return _RERANK_MODEL
    except Exception as _e:
        import sys as _sys
        print(f"[m125_c_rerank] load failed: {_e}; rerank disabled",
              file=_sys.stderr)
        _RERANK_LOAD_FAILED = True
        return None


def cross_encoder_rerank(query, rescored, rerank_k=RERANK_K_DEFAULT):
    """Re-score top-K by cross-encoder, re-sort descending.

    Operates in place on the head of `rescored`. Returns the modified list.
    No-op if model unavailable or pool smaller than 2.
    """
    if not rescored or len(rescored) < 2:
        return rescored
    model = _get_rerank_model()
    if model is None:
        return rescored
    k = min(rerank_k, len(rescored))
    head = rescored[:k]
    pairs = [(query, (r.get("text") or "")[:512]) for r in head]
    try:
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    except Exception as _e:
        import sys as _sys
        print(f"[m125_c_rerank] predict failed: {_e}", file=_sys.stderr)
        return rescored
    for r, s in zip(head, scores):
        r["rerank_score"] = float(s)
        r["reranked"] = True
    # Stable re-sort by rerank_score desc; ties broken by original fused order.
    head.sort(key=lambda r: (-r.get("rerank_score", 0.0),
                             -r.get("fused_score", 0.0)))
    rescored[:k] = head
    return rescored


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
def multi_path_recall(query, store, n_results=5, candidate_pool=30, verbose=False,
                       context_boost=None, rerank=RERANK_ENABLED_DEFAULT,
                       rerank_k=RERANK_K_DEFAULT):
    """Main 35 +1 Task 4: optional context_boost adds active-topic uplift.

    context_boost is a {topic: weight} dict from
    orion-ane/agent/context_tracker.py:ContextTracker.get_retrieval_boost().
    Memories whose metadata.topic matches an active topic get a multiplicative
    score uplift up to 30%. Dormant topics are NOT penalized — they just
    don't get the boost. So a strong base-score memory from a dormant topic
    still surfaces.
    """
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
    # M54 Phase 2.4: possessive-intent detection — distinguishes
    # CAPABILITY questions ("do we have X", "our pipeline") from
    # KNOWLEDGE questions ("what do we know about X"). Knowledge
    # questions should surface external research; capability
    # questions should not. (M53 P4 original was too greedy on
    # "do we" — caught "do we know" as capability.)
    q_low = query.lower()
    capability_markers = [
        "our ", "we have", "we've", "do we have", "do we use",
        "did we build", "did we ship", "did we deploy", "are we using",
        "are we running", "have we built", "have we deployed",
    ]
    knowledge_markers = [
        "do we know", "what do we know", "have we researched",
        "have we explored", "have we investigated", "have we read",
        "what have we found", "have we documented", "have we studied",
    ]
    has_capability = any(m in q_low for m in capability_markers)
    has_knowledge = any(m in q_low for m in knowledge_markers)
    possessive = has_capability and not has_knowledge
    try:
        cosine_results = store.recall(query, n_results=effective_pool,
                                       possessive_intent=possessive)
    except TypeError:
        # Backward compat if store.recall doesn't accept the kwarg
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
            # M97 Fix 2 — domain gate. Canonicals are tagged at inject time
            # with topic ∈ DEFAULT_TOPIC_KEYWORDS (or "universal"). Boost only
            # when the query itself (not the carried-forward active_topics)
            # mentions the canonical's domain, or the canonical is universal.
            # Per-query classification avoids over-boosting on stale context
            # when the current turn is a disambiguation/meta question that
            # doesn't actually touch the prior topic.
            # M125.1 A — atomic-row cosine-dominant scorer. `type='canonical_atom'`
            # rows (A3 atomic ingestion) are sentence-level factual records with
            # empty entities/impacts fields — the structured-signal portion of
            # fused (entity+type+impact = 0.55 weight) always returns 0, so
            # fused ≈ 0.35*s_emb + 0.10*s_rec lands at 0.20-0.28 for typical
            # cosine 0.30-0.40 matches, below the 0.40 midas_ui filter threshold
            # even with CANONICAL_BOOST 1.30. Natural-phrasing queries like
            # "cold prefill time in Main 25" also miss M97 domain-gate keywords
            # → no boost at all → 0.23 measured for 4762 ms row (M125.1 A
            # diagnosis). Override with a cosine-dominant scorer (mirrors the
            # claude_automemory path, line below), bypassing the structured
            # floor and domain gate. Atomic records are specific enough that
            # embedding match is authoritative; no over-boost risk.
            is_atomic = (meta.get("type") or "") == "canonical_atom"
            if is_atomic:
                fused = (0.85 * s_emb + 0.15 * s_rec) * CANONICAL_ATOM_BOOST
            else:
                # M97 Fix 2 — domain gate for coarser architectural canonicals.
                # Canonicals are tagged at inject time with topic ∈
                # DEFAULT_TOPIC_KEYWORDS (or "universal"). Boost only when the
                # query itself (not the carried-forward active_topics) mentions
                # the canonical's domain, or the canonical is universal.
                # Per-query classification avoids over-boosting on stale context
                # when the current turn is a disambiguation/meta question that
                # doesn't actually touch the prior topic.
                canonical_topic = (meta.get("topic") or "").strip().lower()
                query_topics = _m97_classify_query_topics(query)
                if (canonical_topic in ("", "universal")
                        or canonical_topic in query_topics):
                    fused *= CANONICAL_BOOST_ACTIVITY if activity else CANONICAL_BOOST
                # else: no canonical boost on domain mismatch
        elif sr == "meta":
            fused *= META_BOOST_ACTIVITY if activity else META_BOOST
        elif sr == "claude_automemory":
            # Curated finding files (~/.claude/projects/-Users-midas/memory/finding_*.md)
            # — highest confidence, outrank canonical when both exist on a topic.
            # Structural fix (Main 33b): claude_automemory entries are ingested
            # with empty entities/impacts, so the entity/type/impact signals
            # always return 0 and the multiplicative boost can't overcome a
            # canonical that scores well on structured signals. Override to a
            # cosine-dominant scorer that ignores the structured-signal floor.
            fused = (0.85 * s_emb + 0.15 * s_rec) * CLAUDE_AUTOMEMORY_BOOST
        elif sr == "claude_vault_realtime":
            # M125.2 B — class-specific boost (Pattern 2).
            # Parent-synthesis long-body records (M116/M117/M118 etc.) have
            # empty structured-signal fields identical to claude_automemory,
            # so the multiplicative CLAUDE_VAULT_REALTIME_BOOST=1.20 is too
            # weak to clear short canonical rows boosted by CANONICAL_BOOST=1.30
            # riding on structured matches. Swap in a cosine-dominant scorer
            # gated by _is_parent_synthesis (length + source_role + id/file
            # match; all three required). Non-parent-synthesis realtime
            # records (session summaries written to vault, generic agent
            # reports) keep the original 1.20× boost unchanged.
            if _is_parent_synthesis(meta, text):
                fused = (0.85 * s_emb + 0.15 * s_rec) * PARENT_SYNTHESIS_BOOST
            else:
                # Realtime vault enricher ingests — moderate boost.
                fused *= CLAUDE_VAULT_REALTIME_BOOST
        elif sr == "research_ingest":
            # Pre-extraction file dumps — penalize so they don't outrank facts.
            fused *= RESEARCH_INGEST_PENALTY

        # Main 43 Phase 3: topic-matched score boost. 1.5x multiplier for
        # memories matching the active topic. No hard filtering — cross-topic
        # synthesis must still work. Replaces Main 35's +30% max.
        if context_boost:
            mem_topic = (meta.get("topic") or "").strip()
            if mem_topic and mem_topic in context_boost:
                fused *= 1.5

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
            "context_boosted": bool(context_boost and (meta.get("topic") or "").strip() in (context_boost or {})),
        })

    # M122 Stream A4 — narrow ranking boost. Pre-sort, within-pool only.
    # When the query asks for a specific/measurable value, add a constant
    # boost to records that actually contain detail-shape payload
    # (numeric, path, quoted). Under-fit by design; if no detail-shape
    # records exist in the pool, this is a no-op.
    _a4_specific = is_a4_specific_shape_query(query)
    if _a4_specific:
        for r in rescored:
            if is_a4_detail_record(r.get("text", "")):
                r["fused_score"] = r["fused_score"] + A4_BOOST_MAGNITUDE
                r["a4_detail_boosted"] = True
            else:
                r["a4_detail_boosted"] = False
    else:
        for r in rescored:
            r["a4_detail_boosted"] = False
    # Record classifier outcome on every result for downstream diagnostics
    # (ζ v2.2 schema fields may pick this up; emission is A3-scope).
    for r in rescored:
        r["a4_specific_shape_query"] = _a4_specific

    rescored.sort(key=lambda r: r["fused_score"], reverse=True)

    # M125.1 C — cross-encoder rerank. Fires only on specific-shape queries
    # (_a4_specific above). Re-scores top-K with full query-document attention
    # and re-sorts. Under-fit by design — ~22 ms CPU overhead only hits turns
    # that need detail retrieval (~5-10% of traffic on the M125 C2+C3 gold set).
    if rerank and _a4_specific:
        cross_encoder_rerank(query, rescored, rerank_k=rerank_k)
        for r in rescored:
            r.setdefault("reranked", False)
    else:
        for r in rescored:
            r["reranked"] = False

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
def _format_memory_line(m):
    """M122 A1 helper: format a single memory into its `- [tag] text` line.
    Returns (line, text) or (None, None) when the memory has no text body.
    Extracted from the original present() loop body so the canonical-reserve
    path can share identical prefix/provenance logic.
    """
    if isinstance(m, str):
        text = m.strip()
        sr = ""
        session = ""
        topic = ""
    else:
        text = (m.get("text") or "").strip()
        meta = m.get("metadata", {}) or {}
        sr = meta.get("source_role") or m.get("source_role", "")
        session = meta.get("session") or m.get("session", "")
        topic = meta.get("topic") or m.get("topic", "")
    if not text:
        return None, None
    prov_parts = []
    if session:
        import re as _re
        date_m = _re.search(r'(\d{4})-(\d{2})-(\d{2})', session)
        if date_m:
            _months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            mo = int(date_m.group(2))
            prov_parts.append(f"{_months[mo]} {date_m.group(3)}")
        elif "canonical" in session:
            prov_parts.append("canonical")
    if topic:
        prov_parts.append(topic.replace("_", " "))
    if prov_parts:
        prefix = f"[{' / '.join(prov_parts)}] "
    elif sr == "canonical":
        prefix = "[canonical] "
    else:
        prefix = ""
    line = f"  - {prefix}{text}"
    return line, text


# M122 A1: canonical-reserve slot for present(). The highest-scoring canonical
# (source_role="canonical" or role_weight>=1.3) in the recall pool gets a
# dedicated chars budget so it always renders, even when summary-level records
# dominate top-K and consume max_chars. Empirically closes T51/T68 on
# sess_20260421_202025_96397 where canonical-at-pos-5 was truncated.
CANONICAL_RESERVE_CHARS = 400
CANONICAL_ROLE_WEIGHT_FLOOR = 1.3

# M125 A1.2 — canonical-reserve slot widening. Extend from N=1 (M122) to
# N=2 reserve slots. Addresses M124 Stream A Cause 3 (pool cutoff, 6 turns):
# when multiple canonicals exist in the pool but are ranked below non-canonical
# records on cosine, only the top-1 reserved. For multi-session synthesis
# queries (T28 "M115-M118 fix surface", T66 "M108-M121 walkthrough") more
# than one canonical must render to answer correctly.
#
# K2 discipline: cap N=2 to avoid starving the main loop. Third and later
# canonicals fall through to the main-loop ordering. per_query_block rides
# in user-message tail per M122 A1 — prefix cache unaffected (0% delta).
CANONICAL_RESERVE_N = 2

# m122_a3_zeta_v22: side-channel for per-query truncated items. present()
# resets this at entry and populates one entry per pool memory dropped by
# the max_chars budget. Schema:
#   {pos, score, source_role, truncated_chars, would_render_if_budget_raised}
# Callers (midas_ui._build_per_query_block) read via get_last_truncated_items()
# immediately after present() to populate ζ v2.2 field
# retrieval.per_query_truncated_items. A1 stubs here; fuller heuristics for
# `would_render_if_budget_raised` can be refined by the A1 agent owner.
_LAST_TRUNCATED_ITEMS: list = []


def get_last_truncated_items():
    """Return the truncation manifest from the most recent present() call.

    Each entry is a dict: {pos, score, source_role, truncated_chars,
    would_render_if_budget_raised}. Empty list when nothing truncated.
    Returns a shallow copy so callers cannot mutate internal state.
    """
    return list(_LAST_TRUNCATED_ITEMS)


def _memory_source_role(m):
    """Extract source_role string from either nested- or flat-shape memory."""
    if not isinstance(m, dict):
        return ""
    meta = m.get("metadata", {}) or {}
    return meta.get("source_role") or m.get("source_role", "") or ""


def _memory_score(m):
    """Extract score from memory record; 0.0 on failure."""
    if not isinstance(m, dict):
        return 0.0
    try:
        return float(m.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def present(memories, query, max_chars=1200):
    """Format the top-N memories for synthesizer injection, tailored to the
    query category. Accepts both nested-metadata shape (from multi_path_recall
    directly) and flat shape (from MemoryBridge.recall after Main 24 Build 0
    wiring, which lifts source_role to the top level).

    M122 A1: canonical-reserve. Before the main truncation loop iterates,
    scan the pool for the highest-scoring canonical-source-role record. If
    present and not already the top-1 (which would render naturally), reserve
    up to CANONICAL_RESERVE_CHARS out of max_chars for it so it always renders
    even when summary-level records dominate top-K. Preserves original
    ordering for all other items. Closes the T51/T68 synthesis-residual
    mechanism diagnosed in M121 A.
    """
    # m122_a3_zeta_v22: reset truncation manifest for this call.
    _LAST_TRUNCATED_ITEMS.clear()
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

    # M122 A1 / M125 A1.2: identify canonical-reserve indices. Pick up to
    # CANONICAL_RESERVE_N highest-scoring canonicals that would NOT render
    # naturally under a no-reserve pass (i.e. summary-level records would
    # consume the budget first). If every canonical already fits naturally,
    # skip reserve. This preserves the T51/T68 fix (specific canonical at
    # pool pos 4-6 renders despite summary-level records dominating) and
    # extends to multi-canonical synthesis queries (M124 A Cause 3: T66
    # "M108-M121 walkthrough", T28 "M115-M118 fix surface").
    #
    # Simulation: pre-walk the ordered pool, count chars each record would
    # consume, identify which canonicals would get truncated.
    reserved_lines: list[tuple[int, str]] = []  # (canonical_idx, rendered_line)
    remaining_budget = max_chars
    skip_idxs: set[int] = set()

    # Dry-run pass: identify which indices would render under a plain
    # max_chars budget, then pick the highest-scoring canonicals NOT in that
    # set as reserve candidates.
    sim_used = len(header)
    naturally_rendered = set()
    for i, m in enumerate(memories):
        line, _ = _format_memory_line(m)
        if line is None:
            continue
        if sim_used + len(line) > max_chars:
            break
        naturally_rendered.add(i)
        sim_used += len(line) + 1

    # M125 A1.2: collect top-N canonical candidates by score (descending),
    # excluding those that would render naturally. N=2 by design (K2 cap).
    canonical_candidates: list[tuple[float, int]] = []
    for i, m in enumerate(memories):
        if i in naturally_rendered:
            continue
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata", {}) or {}
        sr = meta.get("source_role") or m.get("source_role", "")
        rw = m.get("role_weight")
        if rw is None:
            rw = meta.get("role_weight", 1.0)
        try:
            rw = float(rw)
        except (TypeError, ValueError):
            rw = 1.0
        is_canonical = (sr == "canonical") or (rw >= CANONICAL_ROLE_WEIGHT_FLOOR)
        if not is_canonical:
            continue
        try:
            sc = float(m.get("score", 0.0))
        except (TypeError, ValueError):
            sc = 0.0
        canonical_candidates.append((sc, i))

    # Sort by score descending, take top-N.
    canonical_candidates.sort(key=lambda x: -x[0])
    canonical_candidates = canonical_candidates[:CANONICAL_RESERVE_N]

    # Reserve slots: for each candidate, verify there's enough budget for
    # (header + first-regular-line + reserves + 20-char floor). K2 discipline:
    # stop reserving once adding the next would starve the main loop.
    MAIN_LOOP_FLOOR_CHARS = 20  # main loop must always have some headroom
    for _score, cidx in canonical_candidates:
        m = memories[cidx]
        line, _ = _format_memory_line(m)
        if line is None:
            continue
        if len(line) > CANONICAL_RESERVE_CHARS:
            line = line[:CANONICAL_RESERVE_CHARS - 1].rstrip() + "…"
        reserve_used = len(line) + 1  # +1 newline separator
        already_reserved = sum(len(l) + 1 for _, l in reserved_lines)
        # Budget check: header + already-reserved + this + floor must fit
        if (reserve_used + already_reserved + len(header) +
                MAIN_LOOP_FLOOR_CHARS) > max_chars:
            break  # K2: stop reserving to avoid starving main loop
        reserved_lines.append((cidx, line))
        skip_idxs.add(cidx)
        remaining_budget -= reserve_used

    # Compute a stable insert-position schedule. M122 A1 inserted the sole
    # canonical at position 2 (after top-1 ranked record). Extend: N=2 case
    # inserts reserved[0] at pos 2 and reserved[1] at pos 3.
    # reserved_insert_targets maps `rendered` count -> list of reserved
    # lines to insert at that point.
    reserved_schedule: dict[int, list[str]] = {}
    for slot_idx, (_cidx, line) in enumerate(reserved_lines):
        insert_after = 1 + slot_idx  # first reserve at 1, second at 2, etc.
        reserved_schedule.setdefault(insert_after, []).append(line)

    rendered = 0
    # m122_a3_zeta_v22: track which pool indices rendered so we can emit
    # the truncation manifest for those that did not.
    _rendered_indices = set()
    _truncate_pos_first = None  # first index rejected by the budget
    for i, m in enumerate(memories):
        if i in skip_idxs:
            continue
        line, text = _format_memory_line(m)
        if line is None:
            continue
        if used + len(line) > remaining_budget:
            if _truncate_pos_first is None:
                _truncate_pos_first = i
            break
        lines.append(line)
        used += len(line) + 1
        rendered += 1
        _rendered_indices.add(i)  # m122_a3_zeta_v22
        # Inject any reserved canonical(s) scheduled at this rendered count.
        if rendered in reserved_schedule:
            for rline in reserved_schedule.pop(rendered):
                lines.append(rline)
                used += len(rline) + 1
        # If all reserves have been placed, we can restore remaining_budget
        # to max_chars since any under-budget on reserves returns to main loop.
        if not reserved_schedule:
            remaining_budget = max_chars
    # m122_a3_zeta_v22: mark skipped-but-reserved indices as rendered so
    # they don't appear in the truncation manifest.
    for cidx in skip_idxs:
        _rendered_indices.add(cidx)

    # T51 edge case safeguard: if any reserved canonical never got inserted
    # (e.g. the main loop couldn't render enough records to reach the
    # insert-after checkpoint), emit remaining reserves at the tail so the
    # per-query block is not missing its canonicals.
    for _insert_after, rlines in reserved_schedule.items():
        for rline in rlines:
            lines.append(rline)

    # m122_a3_zeta_v22: populate per_query_truncated_items manifest.
    # Every pool memory with text that did NOT render is a truncation event.
    # `would_render_if_budget_raised` is a simple scaffold heuristic (A1 may
    # refine): True for any item that followed `_truncate_pos_first` in the
    # pool AND has non-empty text — i.e. items that would have rendered
    # sequentially if the budget were larger.
    for i, m in enumerate(memories):
        if i in _rendered_indices:
            continue
        line, _text = _format_memory_line(m)
        if line is None:
            continue  # empty-text memory is not a truncation event
        truncated_chars = len(line)
        would_render = (
            _truncate_pos_first is not None and i >= _truncate_pos_first
        )
        _LAST_TRUNCATED_ITEMS.append({
            "pos": i,
            "score": _memory_score(m),
            "source_role": _memory_source_role(m),
            "truncated_chars": truncated_chars,
            "would_render_if_budget_raised": bool(would_render),
        })

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
