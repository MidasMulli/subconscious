#!/usr/bin/env python3
"""
Subconscious 1C: 1B entity enrichment via ANE cold loop.

Runs after 70B starts responding. Takes CPU-extracted memories
with regex entities. Sends each to 1B on ANE (:8423) for richer
entity extraction. Merges with regex entities (union, deduplicated).

Copyright 2026 Nick Lo. MIT License.
"""

import json
import os
import time
import urllib.request
import logging

import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ENRICH] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("enricher")

ANE_URL = "http://localhost:8423/analyze"
DB_PATH = os.path.expanduser("~/Desktop/cowork/orion-ane/memory/chromadb_live")

ENTITY_PROMPT = 'List the named entities in this text. Output a JSON array of entity names only. Include: model names, hardware, software tools, repos, protocols. Use short names (ANE not Apple Neural Engine). Output JSON array only.\n\nText: {text}\n\nJSON:'


def extract_entities_1b(text, max_tokens=60):
    """Call 1B on ANE for entity extraction."""
    prompt = ENTITY_PROMPT.format(text=text[:300])
    payload = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(ANE_URL, data=payload,
        headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        result = data.get("result", "")
        # Parse JSON array from result
        start = result.find("[")
        end = result.rfind("]")
        if start >= 0 and end > start:
            entities = json.loads(result[start:end+1])
            return [str(e) for e in entities if isinstance(e, str) and len(e) > 1]
    except Exception as e:
        log.debug(f"1B entity extraction failed: {e}")
    return []


def enrich_recent_memories(n_recent=50):
    """Enrich the most recent N memories with 1B entity extraction.

    Only enriches memories that have regex-only entities (no '1b_enriched' flag).
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_collection("conversation_memory")

    # Get recent memories
    all_data = col.get(include=["documents", "metadatas"])
    n = len(all_data['ids'])
    recent_start = max(0, n - n_recent)

    enriched = 0
    for i in range(recent_start, n):
        fid = all_data['ids'][i]
        meta = all_data['metadatas'][i]
        doc = all_data['documents'][i]

        # Skip if already enriched or superseded
        if meta.get('1b_enriched') or meta.get('superseded_by'):
            continue

        # Skip very short memories
        if len(doc) < 20:
            continue

        # Get 1B entities
        new_entities = extract_entities_1b(doc)
        if not new_entities:
            # Mark as enriched (tried, nothing found)
            meta['1b_enriched'] = 'empty'
            col.update(ids=[fid], metadatas=[meta])
            continue

        # Merge with existing entities
        existing = json.loads(meta.get('entities', '[]'))
        if isinstance(existing, str):
            existing = [existing]
        merged = sorted(set(existing + new_entities))

        meta['entities'] = json.dumps(merged)
        meta['1b_enriched'] = 'done'
        col.update(ids=[fid], metadatas=[meta])
        enriched += 1

        if enriched % 10 == 0:
            log.info(f"  Enriched {enriched} memories...")

    log.info(f"Entity enrichment: {enriched}/{n_recent} memories enriched")
    return enriched


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of recent memories")
    args = parser.parse_args()

    # Check ANE server
    try:
        urllib.request.urlopen("http://localhost:8423/health", timeout=2)
    except Exception:
        log.error("ANE server not running on :8423")
        exit(1)

    enrich_recent_memories(args.n)
