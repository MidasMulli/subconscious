#!/usr/bin/env python3
"""
Subconscious memory assembler.

Takes a conversation turn, splits into claims, classifies each,
extracts entities, outputs structured memories.

Integration: after each 70B response, extractor/ processes the
conversation turn and writes memories to the store.

Copyright 2026 Nick Lo. MIT License.
"""

import re
import json
import time
from claim_splitter import split_claims
from rule_classifier import classify


# Known entities for regex extraction
KNOWN_ENTITIES = [
    'ANE', 'GPU', 'AMX', 'CPU', 'CoreML', 'Metal', 'PyTorch',
    'GPT-2', 'Llama', 'Llama-1B', 'Llama-8B', 'Llama-3B',
    'ane-compiler', 'ane-dispatch', 'ane-toolkit',
    'Subconscious', 'SRAM', 'DRAM', 'SLC', 'IOSurface',
    'NeuralNetworkBuilder', 'MIL', 'SwiGLU', 'GELU', 'SiLU',
    'RMSNorm', 'RoPE', 'GQA', 'QKV', 'FFN', 'lm_head',
    'aned', 'kext', 'espresso', 'ChromaDB', 'MLX',
    'Llama-70B', '70B', '1B', '8B', '3B',
    'ct.predict', 'coremltools', 'Accelerate',
]


def extract_entities(text):
    """Regex entity extraction from a claim segment."""
    entities = set()
    text_lower = text.lower()

    for k in KNOWN_ENTITIES:
        if k.lower() in text_lower:
            entities.add(k)

    # Numbers with units
    for m in re.finditer(r'(\d+\.?\d*)\s*(tok/s|GB/s|ms|GB|MB|TFLOPS|us|µs)', text):
        entities.add(f"{m.group(1)} {m.group(2)}")

    return sorted(entities)


def extract_memories(conversation_text):
    """Extract structured memories from a conversation turn.

    Args:
        conversation_text: raw text with Human:/Assistant: markers

    Returns:
        list of memory dicts
    """
    claims = split_claims(conversation_text)

    memories = []
    for claim in claims:
        domain, memory_type = classify(claim['text'])
        entities = extract_entities(claim['text'])

        memories.append({
            'content': claim['text'],
            'memory_type': memory_type,
            'entities': entities,
            'domain': domain,
            'confidence': 'high' if claim['speaker'] == 'Human' else 'medium',
            'timestamp': time.time(),
        })

    return memories


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = """Human: GPT-2 achieves 135.9 tok/s at 37 dispatches on ANE.

A: That's 200us per dispatch round-trip, 2.2x the 93us floor."""

    memories = extract_memories(text)
    print(json.dumps(memories, indent=2, default=str))
