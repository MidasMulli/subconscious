#!/usr/bin/env python3
"""
Subconscious memory maintenance loops.

Three functions, all pure CPU, no model needed:
  1. decay_scores — reduce relevance for unaccessed memories
  2. consolidate_duplicates — merge >0.90 similarity pairs
  3. resolve_contradictions — supersede old conflicting measurements

Run hourly via launchd or cron.

Usage:
    python maintenance.py              # run all three
    python maintenance.py --decay      # decay only
    python maintenance.py --consolidate
    python maintenance.py --contradict

Copyright 2026 Nick Lo. MIT License.
"""

import argparse
import json
import os
import re
import time
import logging
from datetime import datetime

import sys as _sys
_sys.path.insert(0, os.path.expanduser("~/Desktop/cowork/orion-ane/memory"))
import numpy as np

DB_PATH = os.path.expanduser(
    "~/Desktop/cowork/orion-ane/memory/chromadb_live")
COLLECTION = "conversation_memory"

# Decay config
DECAY_HALF_LIFE_DAYS = 7.0     # relevance halves every 7 days
MIN_RELEVANCE = 0.05           # below this, memory is effectively dormant

# Consolidation config
CONSOLIDATION_THRESHOLD = 0.85  # >0.85 similarity = likely duplicate (was 0.92, too strict)

# Contradiction config
CONTRADICTION_LOW = 0.70
CONTRADICTION_HIGH = 0.94
MEASUREMENT_PATTERN = re.compile(
    r'\d+\.?\d*\s*(?:tok/s|GB/s|ms|MB|GB|%|dispatch|TFLOPS|GFLOPS)')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MAINT] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("maintenance")


_LOCAL_STORE = None

def get_collection():
    """Return the LocalMemoryStore collection shim (Main 24 Build 0).

    Maintenance loops were originally written against a chromadb collection.
    LocalMemoryStore.collection exposes the same get/query/upsert/update/delete
    surface, so callers don't need to change.
    """
    global _LOCAL_STORE
    if _LOCAL_STORE is None:
        from local_store import LocalMemoryStore
        _LOCAL_STORE = LocalMemoryStore(DB_PATH)
    return _LOCAL_STORE.collection


def decay_scores(col=None):
    """Reduce relevance_score for memories based on time since last access.

    Memories not accessed recently lose relevance. They stay in the store
    but drop below retrieval threshold over time.
    """
    if col is None:
        col = get_collection()

    all_data = col.get(include=["metadatas"])
    now = time.time()
    updated = 0
    decayed_to_dormant = 0

    for fid, meta in zip(all_data['ids'], all_data['metadatas']):
        # Skip already superseded
        if meta.get('superseded_by'):
            continue

        # Get or init relevance_score
        relevance = float(meta.get('relevance_score', 1.0))
        if relevance <= MIN_RELEVANCE:
            continue

        # Calculate decay based on timestamp
        try:
            ts = datetime.fromisoformat(meta.get('timestamp', '')).timestamp()
        except (ValueError, TypeError):
            continue

        age_days = (now - ts) / 86400
        new_relevance = 2 ** (-age_days / DECAY_HALF_LIFE_DAYS)

        # Factor in access count (frequently accessed memories decay slower)
        access_count = int(meta.get('access_count', 0))
        if access_count > 0:
            # Each access adds 0.1 to the score floor, up to 0.5
            access_floor = min(0.5, access_count * 0.1)
            new_relevance = max(new_relevance, access_floor)

        new_relevance = max(new_relevance, MIN_RELEVANCE)

        if abs(new_relevance - relevance) > 0.01:
            meta['relevance_score'] = str(round(new_relevance, 4))
            col.update(ids=[fid], metadatas=[meta])
            updated += 1
            if new_relevance <= MIN_RELEVANCE:
                decayed_to_dormant += 1

    log.info(f"Decay: {updated} updated, {decayed_to_dormant} now dormant")
    return updated, decayed_to_dormant


def consolidate_duplicates(col=None):
    """Find and merge memory pairs with >0.92 similarity.

    Keeps the longer/more specific version. Marks the shorter as consolidated.
    """
    if col is None:
        col = get_collection()

    all_data = col.get(include=["documents", "metadatas", "embeddings"])
    n = len(all_data['ids'])
    if n < 2:
        return 0

    consolidated = 0
    skip_ids = set()

    # Check recent entries against each other (last 200 to limit cost)
    recent_n = min(200, n)
    ids = all_data['ids'][-recent_n:]
    docs = all_data['documents'][-recent_n:]
    metas = all_data['metadatas'][-recent_n:]
    embs = np.array(all_data['embeddings'][-recent_n:])

    # Compute pairwise similarities for recent entries
    if len(embs) > 1:
        sims = embs @ embs.T

        for i in range(len(ids)):
            if ids[i] in skip_ids:
                continue
            if metas[i].get('superseded_by') or metas[i].get('consolidated'):
                continue

            for j in range(i + 1, len(ids)):
                if ids[j] in skip_ids:
                    continue
                if metas[j].get('superseded_by') or metas[j].get('consolidated'):
                    continue

                if sims[i][j] > CONSOLIDATION_THRESHOLD:
                    # Keep the longer one
                    if len(docs[i]) >= len(docs[j]):
                        keep_idx, drop_idx = i, j
                    else:
                        keep_idx, drop_idx = j, i

                    drop_meta = metas[drop_idx].copy()
                    drop_meta['consolidated'] = ids[keep_idx]
                    drop_meta['superseded_by'] = ids[keep_idx]
                    col.update(ids=[ids[drop_idx]], metadatas=[drop_meta])
                    skip_ids.add(ids[drop_idx])
                    consolidated += 1

    log.info(f"Consolidate: {consolidated} duplicates merged")
    return consolidated


def resolve_contradictions(col=None):
    """Find measurement conflicts in the 0.70-0.94 similarity band.

    Same entity, different numbers → newer timestamp wins.
    """
    if col is None:
        col = get_collection()

    all_data = col.get(include=["documents", "metadatas", "embeddings"])
    n = len(all_data['ids'])
    if n < 2:
        return 0

    resolved = 0

    # Only check entries that contain measurements
    measurement_indices = []
    for i in range(n):
        if MEASUREMENT_PATTERN.search(all_data['documents'][i]):
            if not all_data['metadatas'][i].get('superseded_by'):
                measurement_indices.append(i)

    if len(measurement_indices) < 2:
        return 0

    # Limit to most recent 100 measurement entries
    recent = measurement_indices[-100:]
    embs = np.array([all_data['embeddings'][i] for i in recent])

    if len(embs) > 1:
        sims = embs @ embs.T

        for a in range(len(recent)):
            i = recent[a]
            meta_i = all_data['metadatas'][i]
            if meta_i.get('superseded_by'):
                continue

            for b in range(a + 1, len(recent)):
                j = recent[b]
                meta_j = all_data['metadatas'][j]
                if meta_j.get('superseded_by'):
                    continue

                sim = sims[a][b]
                if CONTRADICTION_LOW < sim < CONTRADICTION_HIGH:
                    # Check if they have different measurements
                    nums_i = set(MEASUREMENT_PATTERN.findall(all_data['documents'][i]))
                    nums_j = set(MEASUREMENT_PATTERN.findall(all_data['documents'][j]))

                    if nums_i and nums_j and nums_i != nums_j:
                        # Different measurements — newer wins
                        ts_i = meta_i.get('timestamp', '')
                        ts_j = meta_j.get('timestamp', '')

                        if ts_i > ts_j:
                            old_idx, new_id = j, all_data['ids'][i]
                        else:
                            old_idx, new_id = i, all_data['ids'][j]

                        old_meta = all_data['metadatas'][old_idx].copy()
                        old_meta['superseded_by'] = new_id
                        col.update(ids=[all_data['ids'][old_idx]], metadatas=[old_meta])
                        resolved += 1

    log.info(f"Contradictions: {resolved} resolved (newer timestamp wins)")
    return resolved


def vault_sync(col=None):
    """Sync dead paths from vault/CLAUDE.md into memory store.

    Finds memories that positively reference killed projects and
    auto-supersedes them with the kill reason from the vault.
    """
    if col is None:
        col = get_collection()

    claude_md = os.path.expanduser("~/Desktop/cowork/CLAUDE.md")
    if not os.path.exists(claude_md):
        log.info("Vault sync: CLAUDE.md not found, skipping")
        return 0

    with open(claude_md) as f:
        content = f.read()

    # Parse dead paths table: | Path | Kill Reason |
    dead_paths = []
    in_dead = False
    for line in content.split("\n"):
        if "Dead Paths" in line and "#" in line:
            in_dead = True
            continue
        if in_dead and line.startswith("| ") and "---" not in line and "Path" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                dead_paths.append({"name": parts[0], "reason": parts[1]})
        elif in_dead and line.startswith("#"):
            in_dead = False

    if not dead_paths:
        log.info("Vault sync: no dead paths found")
        return 0

    superseded = 0
    all_data = col.get(include=["documents", "metadatas"])

    for dp in dead_paths:
        name = dp["name"]
        # Extract keywords from the dead path name (first 3 significant words)
        keywords = [w.lower() for w in name.split() if len(w) > 2][:4]
        if not keywords:
            continue

        for idx, (fid, doc, meta) in enumerate(zip(
                all_data["ids"], all_data["documents"], all_data["metadatas"])):
            # Skip already superseded
            if meta.get("superseded_by"):
                continue

            doc_lower = doc.lower()
            # Check if memory references this project positively
            keyword_hits = sum(1 for kw in keywords if kw in doc_lower)
            if keyword_hits < 2:
                continue

            # Check for positive framing (not already a kill/dead reference)
            negative_markers = ["dead", "killed", "parked", "blocked", "failed",
                               "doesn't work", "abandoned", "superseded"]
            is_negative = any(m in doc_lower for m in negative_markers)
            if is_negative:
                continue

            positive_markers = ["shows", "progress", "works", "achiev", "priority",
                              "revival", "promising", "viable", "active", "next step"]
            is_positive = any(m in doc_lower for m in positive_markers)
            if not is_positive:
                continue

            # Supersede this memory
            meta_copy = meta.copy()
            meta_copy["superseded_by"] = f"vault_sync_{dp['name'][:30]}"
            meta_copy["supersede_reason"] = dp["reason"][:200]
            col.update(ids=[fid], metadatas=[meta_copy])
            superseded += 1

    log.info(f"Vault sync: {superseded} stale memories superseded from {len(dead_paths)} dead paths")
    return superseded


def production_state_sync(col=None):
    """Supersede memories that contradict the current Production Critical Path.

    vault_sync handles dead paths (the "do not revisit" table). This handles
    the orthogonal case: memories that describe an OLD production state that
    has since been superseded by a model swap, server change, or version bump.
    Such memories are factually outdated regardless of framing — a memory
    saying "spec decode is lost" or "Llama-3.3-70B is the production verifier"
    should be superseded the moment CLAUDE.md says otherwise.

    Strategy:
      1. Parse the "## Production Critical Path" / "### Services" table from CLAUDE.md.
      2. Build a list of (current_term, superseded_terms) pairs that we know about.
      3. Find memories that mention any superseded_term as a current/active fact
         and supersede them with a pointer to the current term.

    The superseded_terms list is hard-coded for the cases we currently know
    about and is meant to be extended as the production state evolves. This
    is intentional — production-state contamination is the kind of thing that
    benefits from explicit, named entries rather than fuzzy matching.
    """
    if col is None:
        col = get_collection()

    claude_md = os.path.expanduser("~/Desktop/cowork/CLAUDE.md")
    if not os.path.exists(claude_md):
        log.info("Production state sync: CLAUDE.md not found, skipping")
        return 0
    with open(claude_md) as f:
        content = f.read().lower()

    # Each entry: (current_state_string_in_claude_md, [stale_phrases_to_purge])
    # The "current state" must be present in CLAUDE.md right now for us to
    # supersede the stale phrases — otherwise we'd purge memories about the
    # state that's actually current.
    state_transitions = [
        # Main 19: model swap from Llama 3.3-70B to Qwen 2.5-72B as the verifier.
        (
            "qwen2.5-72b",
            [
                "llama-3.3-70b",
                "llama 3.3-70b",
                "llama-3.3 70b",
                "meta-llama-3.1-70b",   # earlier prod
                "llama 3.1-70b",
            ],
        ),
        # Main 21: spec decode restored on Qwen 72B.
        # Anything saying "spec decode is lost" / "spec decode functionality has been lost"
        # is now wrong. Only purge if CLAUDE.md mentions the new server.
        (
            "qwen_spec_decode_server.py",
            [
                "spec decode functionality has been lost",
                "spec decode is currently not optimized",
                "spec decode is currently lost",
                "spec decode has been lost",
                "lost the spec decode",
                "spec decode that was built into the llama server",
                "no spec decode (next session",
                "no spec decode yet",
                "currently not optimized",  # narrower hit, paired with spec context
                "focus is on restoring the spec decode",
                "restoring the spec decode for the qwen",
                "the 1b cpu drafter, which was built into the llama server, is not portable",
                "1b cpu drafter is not portable",
                "1b cpu drafter, which was previously running at 10.3ms/tok, is not compatible",
                "1b cpu drafter, which was previously running at",
                "is not compatible with the qwen model",
                "running via `mlx_lm.server` on port 8899 without",
                "running via mlx_lm.server on port 8899 without",
                "without n-gram or cpu drafter support",
                "loss of spec decode functionality",
                "leading to slower response",
                "spec decode that was lost in the transition",
                "lost in the transition from the llama",
            ],
        ),
        # Main 21: 0.8B CPU drafter killed.
        (
            "n-gram-only spec decode",
            [
                "the qwen 0.8b cpu drafter is currently not in use",
                "the qwen 0.8b cpu drafter is not in use",
                "qwen 0.8b cpu drafter is not portable",
            ],
        ),
        # EAGLE-3 hallucinated integration. Anything claiming EAGLE-3 is
        # "integrated", "stable", or "viable" on the production verifier
        # contradicts the dead-path table.
        (
            "eagle-3 on quantized 70b",  # the dead-paths entry
            [
                "eagle-3 has already been integrated",
                "eagle-3 is already integrated",
                "eagle-3 is stable",
                "eagle-3 is viable",
                "eagle-3 is still viable",
                "confirming the viability of eagle-3",
                "eagle-3 is integrated into",
                "given this existing integration",  # context: EAGLE-3
                "given that eagle-3 has already been integrated",
            ],
        ),
    ]

    # Validate each transition: only run if the "current_state_in_claude_md"
    # actually appears in CLAUDE.md right now. Otherwise we'd purge memories
    # for a state that's no longer current.
    active_transitions = []
    for current, stales in state_transitions:
        if current.lower() in content:
            active_transitions.append((current, stales))

    if not active_transitions:
        log.info("Production state sync: no validated transitions, skipping")
        return 0

    superseded = 0
    all_data = col.get(include=["documents", "metadatas"])

    for current, stale_phrases in active_transitions:
        for fid, doc, meta in zip(
                all_data["ids"], all_data["documents"], all_data["metadatas"]):
            if meta.get("superseded_by"):
                continue
            doc_lower = doc.lower()
            hit = next((p for p in stale_phrases if p in doc_lower), None)
            if hit is None:
                continue
            meta_copy = meta.copy()
            meta_copy["superseded_by"] = f"prod_state_sync_{current[:40]}"
            meta_copy["supersede_reason"] = (
                f"Stale production state. Current is: {current}. Matched stale phrase: {hit[:120]}"
            )
            col.update(ids=[fid], metadatas=[meta_copy])
            superseded += 1

    log.info(f"Production state sync: {superseded} stale-state memories superseded across {len(active_transitions)} transitions")
    return superseded


def run_all():
    """Run all maintenance functions."""
    col = get_collection()
    total = col.count()
    log.info(f"Starting maintenance on {total} memories")

    t0 = time.time()
    d_updated, d_dormant = decay_scores(col)
    c_merged = consolidate_duplicates(col)
    r_resolved = resolve_contradictions(col)
    v_superseded = vault_sync(col)
    p_superseded = production_state_sync(col)
    # Build 1 (Main 22): three-signal semantic supersession.
    # Catches paraphrased stale-state memories that phrase matching misses.
    sem_superseded = 0
    try:
        from semantic_supersede import semantic_supersession
        sem_stats = semantic_supersession(col=col)
        sem_superseded = sem_stats.get("superseded", 0)
    except Exception as e:
        log.warning(f"semantic_supersession failed: {e}")

    # Build 0 (Main 23): canonical-state INJECTION — the other half of state
    # management. Supersession deletes stale state; this loop creates the
    # current state as first-class memories from CLAUDE.md tables. Together
    # they keep the store aligned with CLAUDE.md as the file evolves.
    canonical_inserted = 0
    canonical_updated = 0
    canonical_unchanged = 0
    canonical_noise_superseded = 0
    try:
        from canonical_inject import canonical_state_inject
        ci_stats = canonical_state_inject(col=col)
        canonical_inserted = ci_stats.get("inserted", 0)
        canonical_updated = ci_stats.get("updated", 0)
        canonical_unchanged = ci_stats.get("unchanged", 0)
        canonical_noise_superseded = ci_stats.get("noise_superseded", 0)
    except Exception as e:
        log.warning(f"canonical_state_inject failed: {e}")

    # Build 1 (Main 24): meta-memory writer — parses CLAUDE_session_log.md
    # bullets into first-class memories so the agent can answer activity
    # questions ("what did we ship today?", "tell me about Build 0").
    meta_inserted = 0
    meta_updated = 0
    meta_unchanged = 0
    meta_orphans = 0
    try:
        from meta_memory_inject import meta_memory_inject
        mm_stats = meta_memory_inject(col=col)
        meta_inserted = mm_stats.get("inserted", 0)
        meta_updated = mm_stats.get("updated", 0)
        meta_unchanged = mm_stats.get("unchanged", 0)
        meta_orphans = mm_stats.get("orphans_superseded", 0)
    except Exception as e:
        log.warning(f"meta_memory_inject failed: {e}")

    # Main 25 close (loop 9): vault sweep — surface deliverables on disk
    # that no knowledge file references. Recurring pattern across Main 24 +
    # Main 25: 5 different agents stopped because the work was already done
    # but unwired (MiniLM CoreML, libllama_cpu_ops.dylib, the streaming
    # fix, the Zin pass catalog, the MMIO register survey). This loop
    # writes a dated report to vault/memory/insights/ on every cycle so
    # the next session sees the orphaned work.
    sweep_unref = 0
    sweep_scanned = 0
    try:
        from vault_sweep import vault_sweep
        sw = vault_sweep(write_report=True)
        sweep_unref = sw.get("unreferenced", 0)
        sweep_scanned = sw.get("scanned", 0)
    except Exception as e:
        log.warning(f"vault_sweep failed: {e}")

    elapsed = time.time() - t0

    log.info(f"Done in {elapsed:.1f}s: "
             f"decayed={d_updated} dormant={d_dormant} "
             f"consolidated={c_merged} contradictions={r_resolved} "
             f"vault_sync={v_superseded} prod_state_sync={p_superseded} "
             f"semantic_supersession={sem_superseded} "
             f"canonical_inject={canonical_inserted}+{canonical_updated}/{canonical_unchanged} "
             f"canonical_noise_superseded={canonical_noise_superseded} "
             f"meta_inject={meta_inserted}+{meta_updated}/{meta_unchanged} "
             f"meta_orphans={meta_orphans} "
             f"vault_sweep={sweep_unref}/{sweep_scanned}")

    return {
        "total_memories": total,
        "decayed": d_updated,
        "dormant": d_dormant,
        "consolidated": c_merged,
        "contradictions_resolved": r_resolved,
        "vault_synced": v_superseded,
        "production_state_synced": p_superseded,
        "semantic_superseded": sem_superseded,
        "canonical_inserted": canonical_inserted,
        "canonical_updated": canonical_updated,
        "canonical_unchanged": canonical_unchanged,
        "elapsed_s": round(elapsed, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subconscious memory maintenance")
    parser.add_argument("--decay", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--contradict", action="store_true")
    args = parser.parse_args()

    if args.decay:
        col = get_collection()
        decay_scores(col)
    elif args.consolidate:
        col = get_collection()
        consolidate_duplicates(col)
    elif args.contradict:
        col = get_collection()
        resolve_contradictions(col)
    else:
        result = run_all()
        print(json.dumps(result, indent=2))
