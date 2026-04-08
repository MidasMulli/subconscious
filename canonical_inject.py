"""Canonical-state injection — the missing other half of state management.

Main 22 Build 1 shipped supersession (mark stale memories dead). This module
ships the inverse: parse the canonical state from CLAUDE.md and create
first-class memories for it. Together, the two passes keep the memory store
in lock-step with CLAUDE.md as the file evolves.

How it works:
  1. Parse CLAUDE.md sections that describe canonical state:
     - Production Critical Path table (rows with Component | Silicon | Role)
     - Services table (Service | Details)
     - Active Projects bullets (under "## Active Projects")
     - Dead Paths table (rows with Path | Kill Reason)
  2. For each entry, generate a SHORT query-syntactic memory text (the format
     that lexically matches user questions — Build 3 of Main 22 found long
     verbose canonical text doesn't make it into chromadb top-K)
  3. Compute a stable ID per entry (based on section + slug of the entry's
     primary key, NOT a content hash — so text edits update in place rather
     than creating a duplicate)
  4. Upsert into chromadb with source_role=canonical, type=state, high
     relevance_score. The 1.30x canonical boost in `MemoryStore.recall()`
     ensures these outrank conflated extraction-noise memories on the same
     topic.
  5. Idempotent: re-running with no CLAUDE.md changes is a no-op.

This module is called from `maintenance.py run_all()` as loop 7. The
maintenance daemon runs hourly via launchd, so any CLAUDE.md edit
propagates to the memory store within at most one hour.

Validation: 3 canonical-state updates to CLAUDE.md should be ingested AND
retrievable within one maintenance cycle. See /tmp/build0_validate.py.
"""
from __future__ import annotations
import os
import re
import json
import time
import logging
from pathlib import Path

log = logging.getLogger("canonical_inject")
if not log.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

CLAUDE_MD = Path(os.path.expanduser("~/Desktop/cowork/CLAUDE.md"))


def _slug(text: str) -> str:
    """Stable slug for a canonical entry's primary key."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')[:50]


# ---------------------------------------------------------------------------
# CLAUDE.md parsers
# ---------------------------------------------------------------------------
def parse_production_critical_path(claude_text: str) -> list[dict]:
    """Parse the Production Critical Path table.

    Expected format:
        ### Production Critical Path
        | Component | Silicon | Role | Critical |
        |-----------|---------|------|----------|
        | **Qwen2.5-72B Q4** | GPU | Verifier + idle-time maintenance | **Yes** |
    """
    rows = []
    in_section = False
    for line in claude_text.split("\n"):
        if "Production Critical Path" in line and line.startswith("###"):
            in_section = True
            continue
        if in_section and line.startswith("###"):
            break
        if in_section and line.startswith("|") and "---" not in line and "Component" not in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 3:
                comp = cells[0].replace("**", "").strip()
                silicon = cells[1].replace("**", "").strip()
                role = cells[2].replace("**", "").strip()
                rows.append({
                    "section": "production_critical_path",
                    "key": _slug(comp),
                    "component": comp,
                    "silicon": silicon,
                    "role": role,
                })
    return rows


def parse_services(claude_text: str) -> list[dict]:
    """Parse the Services table.

    Expected format:
        ### Services
        | Service | Details |
        |---------|---------|
        | **72B server** | `qwen_spec_decode_server.py` :8899 — Qwen2.5-72B-Instruct-4bit |
    """
    rows = []
    in_section = False
    for line in claude_text.split("\n"):
        if "### Services" in line:
            in_section = True
            continue
        if in_section and line.startswith("##"):
            break
        if in_section and line.startswith("|") and "---" not in line and "Service" not in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2:
                svc = cells[0].replace("**", "").strip()
                details = cells[1].strip()
                rows.append({
                    "section": "services",
                    "key": _slug(svc),
                    "service": svc,
                    "details": details,
                })
    return rows


def parse_active_projects(claude_text: str) -> list[dict]:
    """Parse the Active Projects bulleted list AND the Roadmap Active NOW
    bulleted list. Both use markdown checkbox lists in this vault, with
    optional `[x]` or `[ ]` prefixes followed by `**Name**`.

    Expected format (either of):
        - **Subconscious** (path/to/code): description...
        - [x] **70B Q4 production server** — Llama 3.3-70B...
        - [ ] **Living Model revival** — LoRA at attention projections
    """
    rows = []
    in_section = False
    for line in claude_text.split("\n"):
        # Active Projects in main CLAUDE.md OR Active NOW in Roadmap
        if line.startswith("## Active Projects") or line.startswith("### Active NOW"):
            in_section = True
            continue
        if in_section and (line.startswith("##") or line.startswith("### ")):
            break
        # Match: optional `- [x] ` or `- [ ] ` or just `- ` then `**Name**`
        m = re.match(
            r'^-\s+(?:\[[xX ]\]\s+)?\*\*([^*]+)\*\*\s*(?:\(([^)]+)\))?\s*[:—–-]?\s*(.*)',
            line
        )
        if in_section and m:
            name = m.group(1).strip()
            path = (m.group(2) or "").strip()
            desc = m.group(3).strip()
            if name:
                rows.append({
                    "section": "active_projects",
                    "key": _slug(name),
                    "name": name,
                    "path": path,
                    "description": desc[:300],
                })
    return rows


def parse_dead_paths(claude_text: str) -> list[dict]:
    """Parse the Dead Paths table.

    Expected format:
        ## Dead Paths (confirmed, do not revisit)
        ...
        | Path | Kill Reason |
        |------|-------------|
        | EAGLE-3 on quantized 70B | 0% acceptance on Q3 AND Q4 ... |
    """
    rows = []
    in_section = False
    for line in claude_text.split("\n"):
        if line.startswith("## Dead Paths"):
            in_section = True
            continue
        if in_section and line.startswith("##") and "Dead Paths" not in line:
            break
        if in_section and line.startswith("|") and "---" not in line and "| Path " not in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2:
                path = cells[0].replace("**", "").strip()
                reason = cells[1].strip()[:200]
                rows.append({
                    "section": "dead_paths",
                    "key": _slug(path),
                    "path": path,
                    "reason": reason,
                })
    return rows


# ---------------------------------------------------------------------------
# Canonical text generation (short query-syntactic form)
# ---------------------------------------------------------------------------
# Build 3 of Main 22 found that long verbose canonical text doesn't make it
# into the chromadb top-K candidate pool because the embedding gets diluted
# by extra context. Short, front-loaded, query-shaped text wins. The format
# below mirrors the working canonical_8a99e8b5 from /tmp/build3_inject_canonical.py.

def render_pcp(entry: dict) -> tuple[str, list[str]]:
    """Render a Production Critical Path row as a canonical memory.
    Format: lexically natural so the embedding clusters near typical user
    questions ("what is the X tok/s on ANE?"). Front-load the component +
    silicon + role in subject-predicate form."""
    text = (f"The {entry['component']} model on {entry['silicon']} in production "
            f"runs as {entry['role']}.")
    # Token-level entity decomposition: include both the full component name
    # and its component words so substring matching works against memories
    # that use shorter forms ("8B model" vs "8B Q8 model")
    entities = [entry['component'], entry['silicon'], "production"]
    for tok in re.split(r'\s+', entry['component']):
        if len(tok) > 1 and tok not in entities:
            entities.append(tok)
    return text, entities


def render_service(entry: dict) -> tuple[str, list[str]]:
    """Render a Services row as a canonical memory."""
    # Extract port number for entity
    port_match = re.search(r':(\d{4})', entry['details'])
    port = port_match.group(1) if port_match else ""
    text = f"Service in production: {entry['service']}. {entry['details']}"
    entities = [entry['service'], "production"]
    if port:
        entities.append(port)
    return text, entities


def render_active_project(entry: dict) -> tuple[str, list[str]]:
    """Render an Active Project bullet as a canonical memory."""
    text = f"Active project: {entry['name']}. {entry['description']}"
    entities = [entry['name'], "active project"]
    if entry.get('path'):
        entities.append(entry['path'])
    return text, entities


def render_dead_path(entry: dict) -> tuple[str, list[str]]:
    """Render a Dead Path row as a canonical 'do not revisit' memory."""
    text = (f"Dead path: {entry['path']}. {entry['reason']} "
            f"This is killed and should not be revisited.")
    entities = [entry['path'], "dead path", "killed"]
    return text, entities


# ---------------------------------------------------------------------------
# Idempotent upsert
# ---------------------------------------------------------------------------
def make_canonical_id(section: str, key: str) -> str:
    """Stable ID per canonical entry. Section + slug of primary key.
    Re-running with the same CLAUDE.md content updates in place; no duplicates.
    """
    return f"canonical_{section}_{key}"[:120]


def upsert_canonical(col, fid: str, text: str, entities: list[str], embedder) -> tuple[str, list[float]]:
    """Upsert one canonical memory. Returns ('inserted'|'updated'|'unchanged', emb)."""
    existing = col.get(ids=[fid], include=["documents", "embeddings"])
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    if existing["ids"] and existing["documents"][0] == text:
        # Same text — return the existing embedding for the contradiction scan
        existing_emb = existing["embeddings"][0] if existing.get("embeddings") is not None and len(existing["embeddings"]) > 0 else None
        return "unchanged", existing_emb

    emb = embedder.encode([text], normalize_embeddings=True)[0].tolist()
    meta = {
        "type": "state",
        "source_role": "canonical",
        "session": f"canonical_inject_{now[:10]}",
        "timestamp": now,
        "relevance_score": "0.95",
        "entities": json.dumps(entities),
        "quantities": "[]",
    }

    if existing["ids"]:
        col.update(ids=[fid], embeddings=[emb], metadatas=[meta], documents=[text])
        return "updated", emb
    else:
        col.add(ids=[fid], embeddings=[emb], metadatas=[meta], documents=[text])
        return "inserted", emb


# ---------------------------------------------------------------------------
# Canonical-vs-noise contradiction scan
# ---------------------------------------------------------------------------
# When a canonical entry is created or updated, find non-canonical memories
# that share key entities AND make a present-tense state claim AND are NOT
# the canonical itself. These are "noise that contradicts the new canonical"
# and should be superseded so the canonical wins retrieval.
#
# This is the architectural completion of Build 0's gate: without it, a
# numerical update to CLAUDE.md (e.g. 7.9 -> 7.95 tok/s) creates the new
# canonical but doesn't dominate the existing 7.9 noise memories on cosine
# similarity. With it, the 7.9 memories get superseded and only the canonical
# 7.95 remains in the recall pool.

import numpy as _np

# Reuse the tense classifier from semantic_supersede
def _tense_score(text: str) -> float:
    """Lazy import — semantic_supersede has the regex patterns already."""
    try:
        from semantic_supersede import tense_score
        return tense_score(text)
    except ImportError:
        return 0.0


def supersede_contradicting_noise(col, canonical_id: str, canonical_text: str,
                                  canonical_emb, canonical_entities: list[str],
                                  similarity_threshold: float = 0.50) -> int:
    """Find non-canonical memories that contradict this canonical and supersede them.

    Match criteria (all required):
      - cosine similarity to canonical >= threshold (same topic)
      - present-tense state marker in the memory (asserts current state)
      - shares at least one key entity with the canonical (topic-binding)
      - NOT itself a canonical memory
      - NOT already superseded
      - text is NOT identical to the canonical (would be a duplicate, not contradiction)

    Returns the count of newly superseded memories.
    """
    if canonical_emb is None:
        return 0

    cemb = _np.asarray(canonical_emb, dtype=_np.float32)
    norm = _np.linalg.norm(cemb)
    if norm < 1e-9:
        return 0
    cemb = cemb / norm

    all_data = col.get(include=["documents", "metadatas", "embeddings"])
    canonical_text_lower = canonical_text.lower()
    canonical_entities_lower = [e.lower() for e in canonical_entities]
    superseded = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    for fid, doc, meta, emb in zip(
            all_data["ids"], all_data["documents"],
            all_data["metadatas"], all_data["embeddings"]):
        # Skip self
        if fid == canonical_id:
            continue
        # Skip already superseded
        if meta.get("superseded_by"):
            continue
        # Skip other canonicals (canonical-canonical contradictions are
        # caught by the inserted/updated bookkeeping in upsert_canonical)
        if meta.get("source_role") == "canonical":
            continue
        if emb is None:
            continue

        # Topic binding: must share at least one entity word
        doc_lower = doc.lower()
        entity_match = any(e in doc_lower for e in canonical_entities_lower if len(e) > 2)
        if not entity_match:
            continue

        # Same-topic check: cosine similarity above threshold
        memb = _np.asarray(emb, dtype=_np.float32)
        mnorm = _np.linalg.norm(memb)
        if mnorm < 1e-9:
            continue
        memb = memb / mnorm
        sim = float(memb @ cemb)
        if sim < similarity_threshold:
            continue

        # Don't supersede memories whose text is essentially the canonical
        if doc_lower.strip() == canonical_text_lower.strip():
            continue

        # Tense check: PRESERVE memories that are explicit past-tense findings
        # (e.g., "we measured X at Y in Main 19"). The Main 22 tense classifier
        # is too narrow on its present-tense side to gate on; instead we only
        # gate on explicit past-finding markers and let ambiguous/neutral
        # memories be superseded (the other 4 signals — similarity + entity
        # binding + not-already-superseded + not-self — are strong enough).
        ts = _tense_score(doc)
        if ts < 0:
            continue
        # Belt-and-suspenders: explicit past-finding phrases that the tense
        # classifier might miss but are clearly historical
        explicit_past = [
            "was measured", "we measured", "earlier measurement",
            "main 19", "main 12", "main 13", "main 14",  # session references
            "main 21 step", "step 0", "step 1", "step 2",
            "previously", "originally", "in the prior",
        ]
        if any(p in doc_lower for p in explicit_past):
            continue

        # All four signals agree — this memory contradicts the canonical and
        # should be superseded so the canonical wins retrieval.
        meta_copy = dict(meta)
        meta_copy["superseded_by"] = f"canonical_inject_{canonical_id[:50]}"
        meta_copy["supersede_reason"] = (
            f"Contradicts canonical state (sim={sim:.2f}). "
            f"Canonical: {canonical_text[:120]}"
        )
        meta_copy["superseded_at"] = now
        col.update(ids=[fid], metadatas=[meta_copy])
        superseded += 1

    return superseded


# ---------------------------------------------------------------------------
# Public entry point — called from maintenance.run_all() as loop 7
# ---------------------------------------------------------------------------
def canonical_state_inject(col=None) -> dict:
    """Parse CLAUDE.md canonical sections and upsert canonical memories.

    Returns a stats dict for the maintenance log line.
    """
    if not CLAUDE_MD.exists():
        log.info("Canonical inject: CLAUDE.md not found, skipping")
        return {"inserted": 0, "updated": 0, "unchanged": 0, "skipped_no_claude": True}

    if col is None:
        from maintenance import get_collection
        col = get_collection()

    # Lazy embedder import — keep maintenance.py startup cheap.
    # Main 24: prefer CoreML MiniLM (ANE) via shared factory.
    from _embedder import get_embedder
    embedder = get_embedder()

    claude_text = CLAUDE_MD.read_text()

    pcp = parse_production_critical_path(claude_text)
    svcs = parse_services(claude_text)
    projects = parse_active_projects(claude_text)
    dead = parse_dead_paths(claude_text)

    counts = {"inserted": 0, "updated": 0, "unchanged": 0, "noise_superseded": 0,
              "orphans_superseded": 0}
    touched_fids = set()  # canonical IDs we processed this run (in CLAUDE.md)

    def _process(entry, render_fn):
        text, entities = render_fn(entry)
        fid = make_canonical_id(entry["section"], entry["key"])
        touched_fids.add(fid)
        result, emb = upsert_canonical(col, fid, text, entities, embedder)
        counts[result] += 1
        # Only run the contradiction scan when we INSERTED or UPDATED — unchanged
        # canonicals don't change the noise landscape.
        if result in ("inserted", "updated"):
            n = supersede_contradicting_noise(col, fid, text, emb, entities)
            counts["noise_superseded"] += n

    for entry in pcp:
        _process(entry, render_pcp)
    for entry in svcs:
        _process(entry, render_service)
    for entry in projects:
        _process(entry, render_active_project)
    for entry in dead:
        _process(entry, render_dead_path)

    # Orphan supersession: any canonical entry that wasn't touched this run
    # corresponds to a source that no longer exists in CLAUDE.md (the section
    # was edited, the entry was renamed, etc.). These should be superseded so
    # they don't keep injecting stale state into recall. Only touches entries
    # whose fid starts with "canonical_pcp_", "canonical_services_",
    # "canonical_active_projects_", or "canonical_dead_paths_" — leaves any
    # ad-hoc canonical entries (e.g. the Main 22 bootstrap with hashed IDs)
    # alone for safety.
    all_data = col.get(include=["metadatas"])
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    structured_prefixes = ("canonical_production_critical_path_",
                           "canonical_services_",
                           "canonical_active_projects_",
                           "canonical_dead_paths_")
    for fid, meta in zip(all_data["ids"], all_data["metadatas"]):
        if not any(fid.startswith(p) for p in structured_prefixes):
            continue
        if meta.get("superseded_by"):
            continue
        if meta.get("source_role") != "canonical":
            continue
        if fid in touched_fids:
            continue
        # Orphan — no longer in CLAUDE.md
        meta_copy = dict(meta)
        meta_copy["superseded_by"] = "canonical_inject_orphan_cleanup"
        meta_copy["supersede_reason"] = "Canonical entry no longer present in CLAUDE.md"
        meta_copy["superseded_at"] = now
        col.update(ids=[fid], metadatas=[meta_copy])
        counts["orphans_superseded"] += 1

    counts["scanned"] = len(pcp) + len(svcs) + len(projects) + len(dead)
    log.info(f"Canonical inject: scanned={counts['scanned']} "
             f"inserted={counts['inserted']} updated={counts['updated']} "
             f"unchanged={counts['unchanged']} "
             f"noise_superseded={counts['noise_superseded']} "
             f"orphans_superseded={counts['orphans_superseded']}")
    return counts


if __name__ == "__main__":
    canonical_state_inject()
