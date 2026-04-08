"""Meta-memory writer (Main 24 Build 1, loop 8).

Parses `vault/CLAUDE_session_log.md` and emits one memory per timestamped
bullet so the agent can answer "what did we ship today?" / "tell me about
Build 0" / "what changed in Main 23?" — questions that the canonical-state
loop can't answer because they're about *activity*, not *current state*.

Schema (one memory per bullet):
  id          : meta_<YYYYMMDD>_<HHMM>_<8-char-hash>
  text        : "[2026-04-07 12:38] Main 23 closed: 4/4 tasks complete..."
  type        : "session_activity"
  source_role : "meta"
  timestamp   : ISO datetime parsed from the bullet
  session     : "session_<YYYY-MM-DD>"
  atom_type   : "session_activity"
  entities    : JSON list of entities pulled from the bullet text

Idempotent:
  • Stable id from (timestamp + content hash) — re-runs upsert in place
  • Bullets removed from CLAUDE_session_log.md become orphans on next run and
    get superseded with reason="meta_inject_orphan_cleanup"
  • Cap on bullets per run to keep retrieval focused on recent activity (we
    keep the most recent N=200 bullets active; older ones are auto-superseded
    so they don't pollute the recall pool — they remain in the SQLite store
    via include_superseded=True if anything ever needs them)

Wired into maintenance.run_all() as loop 8, after canonical_inject (loop 7).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger("meta_memory_inject")
if not log.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

SESSION_LOG_MD = Path(os.path.expanduser(
    "~/Desktop/cowork/vault/CLAUDE_session_log.md"))

# Cap on active meta memories. Older ones get auto-superseded so they don't
# crowd out recent activity in retrieval. 200 ≈ 4-6 weeks of session bullets.
MAX_ACTIVE_BULLETS = 200

# Match a bullet line:  - **[2026-04-07 12:38]** Free-form summary text...
# Both `- **[date time]** text` and `- [date time] text` (some prior sessions
# omit the bold markers).
BULLET_RE = re.compile(
    r'^\s*-\s*\*?\*?\['
    r'(?P<date>\d{4}-\d{2}-\d{2})'
    r'(?:\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?))?'
    r'\]\*?\*?\s*'
    r'(?P<text>.+?)\s*$'
)

# Entities we care about lifting from session bullets. Mirrors KNOWN_ENTITIES
# in multi_path_retrieve but trimmed to project-activity vocabulary.
ENTITY_PATTERNS = [
    r'\bMain\s+\d+(?:\s+Build\s+\d+[a-z]?)?\b',
    r'\bBuild\s+\d+[a-z]?\b',
    r'\bllama-?(?:1B|3\.1-8B|8B|70B|3\.3-70B)\b',
    r'\bqwen-?(?:0\.8B|2\.5-72B|72B|3\.5-0\.8B)\b',
    r'\b(?:GPT-2|MiniLM|EAGLE-?3?|Medusa|Neuron)\b',
    r'\b(?:ANE|GPU|CPU|AMX|SLC|DRAM|SRAM|MLX|CoreML)\b',
    r'\b(?:8B|72B|70B)\s*(?:Q[348]|FP16)?\b',
    r'\b(?:tok/s|GB/s|ms/tok|tflops)\b',
    r'\b(?:subconscious|chromadb|memory_bridge|midas_ui|maintenance|enricher)\b',
    r'\b(?:multi_path|multi-path|canonical_inject|spec\s*decode|drafter|verifier)\b',
    r'\b(?:vault|paper|gold\s*set|hit@\d+)\b',
    r'\b(?:LocalMemoryStore|sqlite|numpy)\b',
    r'\bMain\s+22\s+Build\s+\d+\b',
    r'\bMain\s+23\s+Build\s+\d+\b',
    r'\bMain\s+24\s+Build\s+\d+\b',
]
ENTITY_RES = [re.compile(p, re.IGNORECASE) for p in ENTITY_PATTERNS]


def parse_session_log(text: str) -> list[dict]:
    """Parse all timestamped bullets out of the session log.

    Returns: [{date, time, text, full_line}, ...] in file order (newest first
    in this repo since the log is append-on-top).
    """
    bullets: list[dict] = []
    for line in text.splitlines():
        m = BULLET_RE.match(line)
        if not m:
            continue
        date = m.group("date")
        t = m.group("time") or "00:00"
        # Normalize HH:MM → HH:MM:00
        if t.count(":") == 1:
            t = t + ":00"
        body = m.group("text").strip()
        if not body:
            continue
        bullets.append({
            "date": date,
            "time": t,
            "text": body,
            "full_line": line.strip(),
        })
    return bullets


def extract_entities(text: str) -> list[str]:
    """Pull project-activity entities from a bullet."""
    found: list[str] = []
    seen = set()
    for r in ENTITY_RES:
        for m in r.finditer(text):
            ent = m.group(0).strip().lower()
            ent = re.sub(r'\s+', ' ', ent)
            if ent and ent not in seen:
                seen.add(ent)
                found.append(ent)
    return found[:12]


def make_meta_id(date: str, time_: str, body: str) -> str:
    """Stable id: meta_<YYYYMMDD>_<HHMM>_<8-char content hash>."""
    h = hashlib.sha1(body.encode("utf-8")).hexdigest()[:8]
    d = date.replace("-", "")
    t = time_[:5].replace(":", "")
    return f"meta_{d}_{t}_{h}"


def render_meta_text(bullet: dict) -> str:
    """One short, query-syntactic line.

    Prefixes the bullet with explicit activity anchor words so cosine matches
    activity-shaped queries ("what did we ship", "what changed", "catch me up",
    "tell me about Build X"). The original bullet text lives unchanged after
    the anchor + date.
    """
    return (f"Session work on {bullet['date']} ({bullet['time'][:5]}) — "
            f"shipped, changed, decided: {bullet['text']}")


def upsert_meta(col, fid: str, text: str, date: str, time_: str,
                entities: list[str], embedder) -> str:
    """Upsert one meta memory. Returns 'inserted' | 'updated' | 'unchanged'."""
    existing = col.get(ids=[fid], include=["documents"])
    if existing["ids"] and existing["documents"][0] == text:
        return "unchanged"

    emb = embedder.encode([text], normalize_embeddings=True)[0].tolist()
    iso_ts = f"{date}T{time_}"
    meta = {
        "type": "session_activity",
        "source_role": "meta",
        "session": f"session_{date}",
        "timestamp": iso_ts,
        "atom_type": "session_activity",
        "atom_schema_version": 1,
        "atom_tense": "past",
        "atom_confidence": 1.0,
        "atom_core": text[:140],
        "entities": json.dumps(entities),
        "atom_entities": json.dumps(entities),
        "quantities": "[]",
        "relevance_score": 0.85,
    }

    if existing["ids"]:
        col.update(ids=[fid], embeddings=[emb], metadatas=[meta], documents=[text])
        return "updated"
    col.add(ids=[fid], embeddings=[emb], metadatas=[meta], documents=[text])
    return "inserted"


# ---------------------------------------------------------------------------
# Public entry point — called from maintenance.run_all() as loop 8
# ---------------------------------------------------------------------------
def meta_memory_inject(col=None) -> dict:
    """Parse the session log and upsert meta memories.

    Caps the active set at MAX_ACTIVE_BULLETS most recent. Older meta
    memories are auto-superseded so retrieval stays focused on recent activity.
    """
    if not SESSION_LOG_MD.exists():
        log.info("Meta inject: session log not found, skipping")
        return {"inserted": 0, "updated": 0, "unchanged": 0,
                "skipped_no_log": True}

    if col is None:
        from maintenance import get_collection
        col = get_collection()

    from _embedder import get_embedder
    embedder = get_embedder()

    text = SESSION_LOG_MD.read_text()
    bullets = parse_session_log(text)
    log.info(f"Meta inject: parsed {len(bullets)} bullets from session log")

    # Sort by (date, time) descending so the cap keeps the most recent.
    bullets.sort(key=lambda b: (b["date"], b["time"]), reverse=True)
    active = bullets[:MAX_ACTIVE_BULLETS]

    counts = {"inserted": 0, "updated": 0, "unchanged": 0,
              "orphans_superseded": 0}
    touched_fids: set[str] = set()

    for b in active:
        body = b["text"]
        entities = extract_entities(body)
        rendered = render_meta_text(b)
        fid = make_meta_id(b["date"], b["time"], body)
        touched_fids.add(fid)
        try:
            result = upsert_meta(col, fid, rendered, b["date"], b["time"],
                                 entities, embedder)
            counts[result] += 1
        except Exception as e:
            log.warning(f"upsert {fid} failed: {e}")

    # Orphan cleanup: any meta_* memory that wasn't touched this run is either
    # (a) older than the cap, or (b) was removed from the session log. Either
    # way, supersede it so it stops showing up in retrieval.
    all_data = col.get(include=["metadatas"])
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for fid, meta in zip(all_data["ids"], all_data["metadatas"]):
        if not fid.startswith("meta_"):
            continue
        if meta.get("source_role") != "meta":
            continue
        if meta.get("superseded_by"):
            continue
        if fid in touched_fids:
            continue
        meta_copy = dict(meta)
        meta_copy["superseded_by"] = "meta_inject_orphan_cleanup"
        meta_copy["supersede_reason"] = (
            f"Meta memory aged out of active window "
            f"(cap={MAX_ACTIVE_BULLETS}) or removed from session log"
        )
        meta_copy["superseded_at"] = now
        try:
            col.update(ids=[fid], metadatas=[meta_copy])
            counts["orphans_superseded"] += 1
        except Exception as e:
            log.warning(f"orphan supersede {fid} failed: {e}")

    counts["scanned"] = len(bullets)
    counts["active_window"] = len(active)
    log.info(f"Meta inject: scanned={counts['scanned']} "
             f"active_window={counts['active_window']} "
             f"inserted={counts['inserted']} updated={counts['updated']} "
             f"unchanged={counts['unchanged']} "
             f"orphans_superseded={counts['orphans_superseded']}")
    return counts


if __name__ == "__main__":
    meta_memory_inject()
