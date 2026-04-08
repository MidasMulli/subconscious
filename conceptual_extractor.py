#!/usr/bin/env python3
"""
Subconscious v1: 70B conceptual extraction.

Runs during idle time. Sends recent conversation turns to 70B
for extraction of decisions, pivots, architectural conclusions,
and trajectory memories that the CPU claim_splitter misses.

Copyright 2026 Nick Lo. MIT License.
"""

import json
import os
import sys
import time
import urllib.request
import logging

import sys as _sys
_sys.path.insert(0, os.path.expanduser("~/Desktop/cowork/orion-ane/memory"))
_sys.path.insert(0, os.path.expanduser("~/Desktop/cowork/vault/subconscious"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CONCEPT] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("conceptual")

LLM_URL = "http://127.0.0.1:8899/v1/chat/completions"
LLM_MODEL = "mlx-community/Llama-3.3-70B-Instruct-3bit"
DB_PATH = os.path.expanduser("~/Desktop/cowork/orion-ane/memory/chromadb_live")

EXTRACTION_PROMPT = """Read this conversation segment. Extract high-level memories that a factual claim splitter would miss. Focus on:
- Decisions made ("we decided to park spec decode")
- Pivots ("shifted from lm_head optimization to subconscious")
- Architectural conclusions ("the ANE path is for agent workloads, not spec decode drafting")
- Relationship insights ("Subconscious supplements the vault, never replaces it")
- Project trajectory ("Phase 0 proved CPU extraction works, 1B handles entity enrichment, 70B handles conceptual extraction")

Do NOT extract measurements, numbers, or specific technical facts. The CPU pipeline handles those. You handle the meaning.

Output: JSON array matching this schema. Output ONLY the JSON array, no other text.
Each object has: "content" (string), "memory_type" ("conceptual"), "entities" (array of strings), "domain" (string), "confidence" ("high")

CONVERSATION:
---
$CONVERSATION$
---

JSON:"""


def llm_call(prompt, max_tokens=800):
    """Call 70B for conceptual extraction."""
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "repetition_penalty": 1.1,
    }).encode()
    req = urllib.request.Request(LLM_URL, data=payload,
        headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


def parse_json_array(text):
    """Parse JSON array from 70B output."""
    text = text.strip()
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        # Try extracting individual objects
        objects = []
        depth = 0
        obj_start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0: obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        obj = json.loads(text[obj_start:i+1])
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
        return objects if objects else None

    try:
        data = json.loads(text[start:end+1])
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    return None


def extract_conceptual(conversation_text):
    """Extract conceptual memories from conversation via 70B."""
    prompt = EXTRACTION_PROMPT.replace("$CONVERSATION$", conversation_text)
    raw = llm_call(prompt)
    if not raw:
        return []

    memories = parse_json_array(raw)
    if not memories:
        log.warning("Failed to parse 70B output")
        return []

    # Validate and clean
    valid = []
    for m in memories:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "").strip()
        if len(content) < 10:
            continue
        valid.append({
            "content": content,
            "memory_type": m.get("memory_type", "conceptual"),
            "entities": m.get("entities", []),
            "domain": m.get("domain", "research"),
            "confidence": m.get("confidence", "high"),
            "source": "70b_synthesis",
        })

    return valid


def store_memories(memories, col, emb_model):
    """Store conceptual memories in ChromaDB with dedup."""
    stored = 0
    for mem in memories:
        emb = emb_model.encode([mem["content"]], normalize_embeddings=True,
                                show_progress_bar=False)[0]

        # Dedup: check if very similar entry exists
        if col.count() > 0:
            results = col.query(query_embeddings=[emb.tolist()], n_results=1)
            if results["distances"][0] and (1 - results["distances"][0][0]) > 0.92:
                continue  # too similar to existing

        counter = col.count() + 1
        fid = f"concept_{counter}_{int(time.time())}"
        meta = {
            "type": mem["memory_type"],
            "entities": json.dumps(mem["entities"]),
            "domain": mem["domain"],
            "confidence": mem["confidence"],
            "source": "70b_synthesis",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        col.upsert(ids=[fid], embeddings=[emb.tolist()],
                    documents=[mem["content"]], metadatas=[meta])
        stored += 1

    return stored


def run_on_chunks(chunk_paths):
    """Run conceptual extraction on specified chunk files."""
    from local_store import LocalMemoryStore
    from _embedder import get_embedder
    store = LocalMemoryStore(DB_PATH)
    col = store.collection
    emb_model = get_embedder()

    for path in chunk_paths:
        name = os.path.basename(path).replace("chunk_", "").replace(".txt", "")
        log.info(f"Processing {name}...")

        with open(path) as f:
            text = f.read()

        t0 = time.time()
        memories = extract_conceptual(text)
        extract_time = time.time() - t0
        log.info(f"  Extracted {len(memories)} conceptual memories in {extract_time:.1f}s")

        for m in memories:
            log.info(f"    [{m['domain']}] {m['content'][:80]}...")

        stored = store_memories(memories, col, emb_model)
        log.info(f"  Stored {stored} (deduped {len(memories) - stored})")

    return memories


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", nargs="*",
                        help="Chunk files to process")
    parser.add_argument("--test-c1", action="store_true",
                        help="Run on C1 gold-set chunk")
    args = parser.parse_args()

    if args.test_c1:
        chunks = [os.path.expanduser(
            "~/Desktop/cowork/vault/subconscious/chunks/chunk_C1.txt")]
    elif args.chunks:
        chunks = args.chunks
    else:
        # Default: all chunks
        chunk_dir = os.path.expanduser(
            "~/Desktop/cowork/vault/subconscious/chunks/")
        chunks = sorted([os.path.join(chunk_dir, f)
                        for f in os.listdir(chunk_dir) if f.endswith(".txt")])

    run_on_chunks(chunks)
