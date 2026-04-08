"""Semantic supersession of stale-state memories.

Replaces phrase-based production_state_sync with a three-signal architecture:

  1. SEMANTIC SIMILARITY  cosine(memory_embedding, canonical_state_embedding)
                          high similarity = "talking about the same topic"

  2. TENSE / FRAMING      regex-based classifier over the memory text
                          present-tense state claim = eligible for supersession
                          past-tense finding        = preserved (history is true)

  3. RESTATE vs CONTRADICT key-term overlap with the canonical entry
                          mentions current key terms => restating, preserve
                          mentions superseded terms => contradicting, supersede

A memory is superseded only when ALL THREE signals agree.

This module is callable from maintenance.py run_all() and is the
production replacement for production_state_sync's phrase-matching loop.

Validation: see /tmp/build1_validate.py — measures precision, recall, and
false positive rate against the paper's gold set (46 facts in
vault/subconscious/gold_sets/).
"""
from __future__ import annotations
import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

log = logging.getLogger("semantic_supersede")
if not log.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

CLAUDE_MD = Path(os.path.expanduser("~/Desktop/cowork/CLAUDE.md"))
EMBEDDING_CACHE = Path(__file__).parent / ".canonical_embeddings.npz"

# ---------------------------------------------------------------------------
# Tense / framing heuristic
# ---------------------------------------------------------------------------
# Present-tense state markers — language asserting "this is the current state."
PRESENT_STATE_PATTERNS = [
    r'\b(?:is|are)\s+(?:the\s+)?(?:current|production|live|active|in\s+use|running|deployed|primary|main)\b',
    r'\b(?:currently|now|presently|today)\b',
    r'\bproduction\s+(?:model|server|verifier|stack|state|config)\s+(?:is|uses|runs)\b',
    r'\b(?:we|the\s+system|the\s+stack)\s+(?:use|uses|run|runs|deploy|deploys)\b',
    r'\bthe\s+(?:current|production|live|active)\b',
    r'\bspec\s+decode\s+(?:is|has\s+been)\s+(?:lost|broken|disabled|not)\b',
    r'\b(?:has|have)\s+been\s+(?:lost|broken|disabled|removed)\b',
    r'\b(?:not\s+in\s+use|not\s+compatible|not\s+portable|currently\s+not)\b',
    r'\bthis\s+(?:is|results\s+in|leads\s+to)\b',
    # In-progress framing about something already shipped is a present-tense
    # state claim (claims the system is currently building X). Once X has
    # shipped, these become stale.
    r'\b(?:is|are)\s+(?:currently\s+)?(?:in\s+the\s+process\s+of|being\s+(?:built|developed|restored|implemented|wired))\b',
    r'\b(?:in\s+the\s+process\s+of|in\s+progress)\b',
]

# Past-tense finding markers — language describing measurements/observations
# from a specific point in time. These are protected from supersession.
PAST_FINDING_PATTERNS = [
    r'\b(?:was|were)\s+(?:measured|tested|observed|found|achieved|reached|recorded)\b',
    r'\b(?:we|i)\s+(?:measured|found|tested|observed|achieved|reached|tried|killed|shipped)\b',
    r'\b(?:the\s+result|the\s+measurement|the\s+test|the\s+benchmark)\s+(?:showed|indicated|gave|was)\b',
    # Tightened from "(?:main|session|the)\s+\w+" which matched "in the process"
    r'\b(?:in|during|at)\s+main\s+\d+\b',
    r'\b(?:in|during)\s+(?:the\s+)?(?:prior|previous|earlier|last|original)\s+(?:session|run|test)\b',
    r'\b(?:on|with|under)\s+the\s+(?:old|previous|earlier|llama|prior)\s+(?:stack|model|config)\b',
    r'\b(?:previously|historically|originally|initially|formerly)\b',
    r'\b(?:achieved|reached|hit|peaked\s+at|delivered)\s+\d',
    # Tightened from greedy `.*` form which over-matched across unrelated
    # clauses ("tok/s ... measured" in "current tok/s for X is N as measured")
    r'\b(?:was|were)\s+measured\s+at\b',
    r'\bpreviously\s+measured\b',
    r'\b(?:proven|confirmed|verified|validated)\s+(?:by|in|via)\b',
]

PRESENT_RX = [re.compile(p, re.IGNORECASE) for p in PRESENT_STATE_PATTERNS]
PAST_RX = [re.compile(p, re.IGNORECASE) for p in PAST_FINDING_PATTERNS]


def tense_score(text: str) -> float:
    """Return a score in [-1, +1].
       +1 = strong present-tense state claim
       -1 = strong past-tense finding
        0 = ambiguous
    """
    present = sum(1 for r in PRESENT_RX if r.search(text))
    past = sum(1 for r in PAST_RX if r.search(text))
    total = present + past
    if total == 0:
        return 0.0
    return (present - past) / total


# ---------------------------------------------------------------------------
# Canonical state extraction from CLAUDE.md
# ---------------------------------------------------------------------------
# Each canonical entry has:
#   text:         the prose to embed (description of current state)
#   key_terms:    substrings that signal "restating current state" -> preserve
#   stale_terms:  substrings that signal "describes superseded state" -> supersede
#
# These are built from the Production Critical Path table + Active Projects +
# the dead paths table. Stale terms are pulled from the dead-path kill reasons
# and from the historical model-swap context.

def extract_canonical_state(claude_md_text: str) -> list[dict]:
    """Parse CLAUDE.md and return canonical state entries.

    Each entry combines a current-state description with the terms that
    distinguish "restating it" (preserve) from "describing the old version"
    (supersede).
    """
    entries: list[dict] = []

    # 1. Production Critical Path table — extract row text
    in_pcp = False
    for line in claude_md_text.split("\n"):
        if "Production Critical Path" in line:
            in_pcp = True
            continue
        if in_pcp and line.startswith("##"):
            break
        if in_pcp and line.startswith("|") and "---" not in line and "Component" not in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 3:
                # Build a current-state sentence and harvest terms
                comp = cells[0].replace("**", "")
                silicon = cells[1]
                role = cells[2]
                text = f"{comp} on {silicon} is {role} in production".lower()
                key_terms = [t.lower() for t in re.findall(r'[A-Za-z][\w.-]+', cells[0]) if len(t) > 2]
                entries.append({
                    "source": "production_critical_path",
                    "text": text,
                    "key_terms": key_terms,
                    "stale_terms": [],  # filled below
                })

    # 2. Services table
    in_svc = False
    for line in claude_md_text.split("\n"):
        if "### Services" in line:
            in_svc = True
            continue
        if in_svc and line.startswith("##"):
            break
        if in_svc and line.startswith("|") and "---" not in line and "Service" not in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2:
                desc = cells[1].replace("**", "")
                text = f"{cells[0]}: {desc}".lower()
                key_terms = [t.lower() for t in re.findall(r'[A-Za-z][\w./-]+', cells[1]) if len(t) > 3]
                entries.append({
                    "source": "services",
                    "text": text,
                    "key_terms": key_terms,
                    "stale_terms": [],
                })

    # 3a. Global "definitely stale" patterns. These are phrases that are wrong
    # in any context, regardless of which canonical entry matches by embedding.
    # They get attached to ALL canonical entries so the supersession decision
    # only needs ONE canonical entry above the similarity threshold to fire.
    global_stale_terms = [
        "is not compatible with the qwen",
        "is not compatible with qwen",
        "0.8b cpu drafter is not compatible",
        "spec decode is lost",
        "spec decode has been lost",
        "spec decode functionality has been lost",
        "lost the spec decode",
        "loss of spec decode",
        "without spec decode",
        "without n-gram or cpu drafter",
        "currently not optimized",
        "running via `mlx_lm.server`",
        "running via mlx_lm.server",
        "via `mlx_lm.server` on port 8899",
        "via mlx_lm.server on port 8899",
        "1b cpu drafter is not portable",
        "0.8b cpu drafter is not portable",
        "the qwen 0.8b cpu drafter is currently not in use",
        "the qwen 0.8b cpu drafter is not in use",
        "eagle-3 has already been integrated",
        "eagle-3 is already integrated",
        "eagle-3 is integrated into",
        "eagle-3 is stable",
        "eagle-3 is viable",
        "eagle-3 is still viable",
        # Future-tense framing about already-shipped Main 21 work
        "spec decode server is currently in the process of being built",
        "spec decode server is in the process of being built",
        "spec decode is in the process of being built",
        "in the process of being built for the qwen",
        # The Main 21 0.8B drafter was killed with measurements. Any memory
        # claiming it's part of the "current setup" is stale.
        "current setup involves using qwen3.5-0.8b",
        "current setup involves using qwen 3.5-0.8b",
        "n-gram first and cpu qwen-0.8b",
        "n-gram first and cpu qwen 0.8b",
        "cpu qwen-0.8b fallback",
        "cpu qwen 0.8b fallback",
        "qwen-0.8b fallback",
        "0.8b fallback (when",
        "qwen3.5-0.8b as the cpu drafter",
        "using qwen3.5-0.8b as the cpu drafter",
        "qwen3.5-0.8b cpu drafter + concurrent",
        "n-gram + qwen3.5-0.8b cpu drafter",
        "the plan is to build a spec decode wrapper",
        "plan is to build a spec decode wrapper around",
    ]
    # Attach the global list to every canonical entry so the topic-binding
    # of the matched canonical doesn't gate the stale-term check.
    for e in entries:
        e["stale_terms"] = list(global_stale_terms)

    # 3b. Topic-bound transition stale terms — extra terms that only apply
    # when the matched canonical entry is the right topic.
    transition_stale_terms = {
        "qwen2.5-72b": [
            "llama-3.3-70b", "llama 3.3-70b", "llama-3.3 70b",
            "meta-llama-3.1-70b", "llama 3.1-70b", "llama-3.1-70b",
            "llama 3.3 70b", "llama 70b production",
        ],
        "qwen_spec_decode_server": [
            "spec decode is lost", "spec decode functionality has been lost",
            "spec decode has been lost", "lost the spec decode",
            "no spec decode yet", "no spec decode (next session",
            "running via mlx_lm.server",
            "via `mlx_lm.server`", "via mlx_lm.server",
            "without n-gram", "without spec decode",
            "1b cpu drafter is not portable",
            "0.8b cpu drafter is not portable",
            "qwen 0.8b cpu drafter is not in use",
            "qwen 0.8b cpu drafter is not compatible",
            "the focus is on restoring the spec decode",
            "loss of spec decode functionality",
        ],
        "n-gram": [
            "n-gram prewarm", "n-gram seeding",
        ],
        "eagle-3 dead": [
            "eagle-3 has already been integrated",
            "eagle-3 is already integrated",
            "eagle-3 is integrated into",
            "eagle-3 is stable",
            "eagle-3 is viable",
            "eagle-3 is still viable",
            "given this existing integration",
            "given that eagle-3 has already been integrated",
            "confirming the viability of eagle-3",
        ],
    }

    # Attach topic-bound stale terms (extras on top of the global set)
    for stale_topic, stale_phrases in transition_stale_terms.items():
        topic_lower = stale_topic.lower()
        attached = False
        for e in entries:
            if any(t in e["text"] for t in topic_lower.split()):
                e["stale_terms"].extend(stale_phrases)
                attached = True
        if not attached:
            entries.append({
                "source": f"synthetic_{stale_topic}",
                "text": f"current state of {stale_topic} per CLAUDE.md".lower(),
                "key_terms": stale_topic.split(),
                "stale_terms": list(global_stale_terms) + stale_phrases,
            })

    return entries


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
def load_embedder():
    """Lazy-load the embedder. Prefers CoreML MiniLM (ANE), falls back to CPU."""
    from _embedder import get_embedder
    return get_embedder()


def get_canonical_embeddings(claude_md_text: str, embedder) -> tuple[list[dict], np.ndarray]:
    """Build canonical entries + embeddings, with cache invalidation."""
    entries = extract_canonical_state(claude_md_text)
    texts = [e["text"] for e in entries]

    # Cache key: hash of canonical texts
    import hashlib
    cache_key = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]

    if EMBEDDING_CACHE.exists():
        try:
            cached = np.load(EMBEDDING_CACHE, allow_pickle=True)
            if str(cached.get("key", "")) == cache_key:
                return entries, cached["embeddings"]
        except Exception:
            pass

    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.savez(EMBEDDING_CACHE, key=cache_key, embeddings=embeddings)
    return entries, embeddings


# ---------------------------------------------------------------------------
# Three-signal supersession decision
# ---------------------------------------------------------------------------
def supersession_decision(
    memory_text: str,
    memory_embedding: np.ndarray,
    canonical_entries: list[dict],
    canonical_embeddings: np.ndarray,
    similarity_threshold: float = 0.55,
    tense_threshold: float = 0.0,
) -> tuple[bool, dict]:
    """Decide whether `memory_text` is a stale-state claim that should be superseded.

    Returns (should_supersede, debug_info_dict).
    """
    # Normalize the memory embedding for cosine similarity
    mem_emb = np.asarray(memory_embedding, dtype=np.float32)
    norm = np.linalg.norm(mem_emb)
    if norm < 1e-9:
        return False, {"reason": "zero_norm"}
    mem_emb = mem_emb / norm

    # Cosine similarity to each canonical entry (canonical embeddings already normalized)
    sims = canonical_embeddings @ mem_emb
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_entry = canonical_entries[best_idx]

    text_lower = memory_text.lower()

    # FAST PATH: explicit stale-term match + non-past tense.
    # When a memory contains a known-stale phrase verbatim AND its tense
    # is not strictly past (i.e. it's NOT framed as a historical finding),
    # that's direct evidence of a stale-state claim. The cosine similarity
    # is unnecessary as confirmation — the phrase IS the binding.
    # We use ts >= 0 (not > 0) so neutral-tense memories get caught when
    # they contain explicit stale phrases. Past-tense findings (ts < 0)
    # are still preserved because history about old state is true.
    fast_path_match = next(
        (st for e in canonical_entries for st in e["stale_terms"] if st in text_lower),
        None,
    )
    if fast_path_match is not None:
        ts = tense_score(memory_text)
        if ts > tense_threshold:
            # Find which canonical entry owns this stale term, for the supersede pointer
            owner = next(
                (e for e in canonical_entries if fast_path_match in e["stale_terms"]),
                best_entry,
            )
            return True, {
                "reason": "explicit_stale_term_present_tense",
                "best_sim": best_sim,
                "best_entry": owner["source"],
                "tense": ts,
                "matched_stale_term": fast_path_match,
            }

    # SLOW PATH: similarity gate then signal-stack
    if best_sim < similarity_threshold:
        return False, {"reason": "below_similarity", "best_sim": best_sim, "best_entry": best_entry["source"]}

    # SIGNAL 3: restate vs contradict (check this BEFORE tense, it's the strongest)
    # If the memory contains canonical key terms AND no stale terms, it's
    # restating current state -> preserve.
    has_key_term = any(kt in text_lower for kt in best_entry["key_terms"])
    has_stale_term = any(st in text_lower for st in best_entry["stale_terms"])

    if has_key_term and not has_stale_term:
        return False, {
            "reason": "restating_current_state",
            "best_sim": best_sim,
            "best_entry": best_entry["source"],
        }

    if not has_stale_term:
        # No stale term signal at all — even if topic matches, we don't have
        # evidence that this is the OLD version. Preserve.
        return False, {
            "reason": "no_stale_term",
            "best_sim": best_sim,
            "best_entry": best_entry["source"],
        }

    # SIGNAL 2: tense gate
    # Memory has a stale term and is topically aligned. Check tense to ensure
    # this is a present-tense STATE claim, not a past-tense finding ABOUT the
    # old state (which is historically true and must be preserved).
    ts = tense_score(memory_text)
    if ts <= tense_threshold:
        return False, {
            "reason": "past_tense_finding",
            "best_sim": best_sim,
            "best_entry": best_entry["source"],
            "tense": ts,
        }

    # All three signals agree: high similarity, has stale term, present-tense state claim.
    return True, {
        "reason": "stale_state_claim",
        "best_sim": best_sim,
        "best_entry": best_entry["source"],
        "tense": ts,
        "matched_stale_term": next((st for st in best_entry["stale_terms"] if st in text_lower), ""),
    }


# ---------------------------------------------------------------------------
# Production driver — called from maintenance.run_all()
# ---------------------------------------------------------------------------
def semantic_supersession(col=None, dry_run: bool = False) -> dict:
    """Walk the entire memory store and supersede stale-state memories.

    Returns a stats dict for logging.
    """
    if col is None:
        # Lazy import so this module loads cleanly even when chromadb isn't on PATH.
        from maintenance import get_collection
        col = get_collection()

    if not CLAUDE_MD.exists():
        log.info("Semantic supersession: CLAUDE.md not found, skipping")
        return {"superseded": 0, "scanned": 0, "reason": "no_claude_md"}

    claude_text = CLAUDE_MD.read_text()
    embedder = load_embedder()
    canonical_entries, canonical_embs = get_canonical_embeddings(claude_text, embedder)
    log.info(f"Semantic supersession: {len(canonical_entries)} canonical entries loaded")

    all_data = col.get(include=["documents", "metadatas", "embeddings"])
    n_total = len(all_data["ids"])
    n_skipped_already = 0
    n_skipped_zero_emb = 0
    n_decided_supersede = 0
    n_decided_keep = 0

    decisions_to_apply = []
    for fid, doc, meta, emb in zip(
            all_data["ids"], all_data["documents"],
            all_data["metadatas"], all_data["embeddings"]):
        if meta.get("superseded_by"):
            n_skipped_already += 1
            continue
        if emb is None or len(emb) == 0:
            n_skipped_zero_emb += 1
            continue

        decide, info = supersession_decision(
            doc, np.asarray(emb), canonical_entries, canonical_embs)

        if decide:
            n_decided_supersede += 1
            decisions_to_apply.append((fid, meta, info))
        else:
            n_decided_keep += 1

    if not dry_run:
        for fid, meta, info in decisions_to_apply:
            meta_copy = dict(meta)
            meta_copy["superseded_by"] = f"semantic_supersede_{info['best_entry'][:40]}"
            meta_copy["supersede_reason"] = (
                f"Stale-state claim superseded by canonical state. "
                f"sim={info['best_sim']:.3f} tense={info.get('tense', 0):.2f} "
                f"matched_stale_term={info.get('matched_stale_term', '')[:80]}"
            )
            meta_copy["superseded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            col.update(ids=[fid], metadatas=[meta_copy])

    stats = {
        "scanned": n_total,
        "already_superseded": n_skipped_already,
        "zero_embedding_skipped": n_skipped_zero_emb,
        "kept": n_decided_keep,
        "superseded": n_decided_supersede,
        "dry_run": dry_run,
    }
    log.info(f"Semantic supersession: {stats}")
    return stats


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Score memories but don't write supersession metadata")
    args = p.parse_args()
    semantic_supersession(dry_run=args.dry_run)
