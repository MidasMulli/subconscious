#!/usr/bin/env python3
"""
Rule-based domain and memory_type classifier for claim segments.

Pure keyword matching. No model. First match wins.

Copyright 2026 Nick Lo. MIT License.
"""

import re


# Domain rules: keyword sets, first match wins
DOMAIN_RULES = [
    ("hardware", [
        r'tok/s', r'dispatch', r'SRAM', r'DRAM', r'bandwidth', r'GB/s',
        r'\bANE\b', r'\bGPU\b', r'\bAMX\b', r'\bMetal\b', r'CoreML',
        r'kext', r'register', r'opcode', r'\.hwx', r'FP16', r'FP32',
        r'INT8', r'GELU', r'SiLU', r'ReLU', r'conv\b', r'FLOPS',
        r'latency', r'\bms\b', r'µs', r'IOSurface', r'aned\b',
        r'dispatch floor', r'tile', r'pipeline', r'compile',
        r'weight', r'activation', r'fusion', r'fused',
        r'PWL', r'espresso', r'\bDMA\b', r'SLC', r'cache',
    ]),
    ("production", [
        r':8899', r':8423', r':8891', r'server', r'daemon',
        r'deploy', r'config\b', r'stack', r'pipeline',
        r'endpoint', r'HTTP', r'socket', r'launchd',
        r'orchestrat', r'verifier', r'drafter',
        r'spec.?decode', r'n-gram', r'production',
        r'Subconscious', r'memory manager', r'ingestion',
        r'retrieval', r'maintenance', r'hot loop', r'cold loop',
    ]),
    ("research", [
        r'paper\b', r'arXiv', r'benchmark', r'finding',
        r'published', r'novel', r'kill test', r'PyTorch',
        r'hypothesis', r'experiment', r'proof', r'verified',
        r'correctness', r'match\b',
    ]),
    ("hardware", [
        r'\bANE\b', r'\bAMX\b', r'\bSME\b', r'\bSLC\b', r'\bDMA\b',
        r'M5 Pro', r'M5\b', r'M4\b', r'\bSRAM\b', r'\bDRAM\b',
        r'\bMCC\b', r'tile\b', r'dispatch floor', r'kext\b',
        r'IOSurface', r'IOKit', r'exclave',
    ]),
    ("personal", [
        r'prefer\b', r'want\b', r'hate\b', r'always\b',
        r'never\b', r'love\b', r'style\b',
    ]),
]

# Memory type rules
TYPE_RULES = [
    ("preference", [
        r'want\b', r'prefer\b', r'hate\b', r'always\b',
        r'never\b', r'should\b', r'dont\b.*do',
    ]),
    ("relationship", [
        r'feeds into', r'within\b', r'part of',
        r'uses\b', r'runs on', r'connects to',
        r'replaces', r'depends on', r'built on',
        r'component', r'consists of',
    ]),
    ("state", [
        r'\d+\.?\d*\s*tok/s', r'\d+\.?\d*\s*ms\b',
        r'\d+\.?\d*\s*GB', r'\d+\.?\d*%',
        r'currently', r'\bnow\b', r'running\b',
        r'achieves', r'measured', r'produces',
        r'passes', r'blocked', r'working\b',
        r'shipped', r'live\b', r'ready\b',
        r'from\s+\d.*to\s+\d', r'up from',
    ]),
    ("fact", [
        r'is a\b', r'\bhas\b', r'consists of',
        r'architecture', r'defined as', r'means\b',
        r'because\b', r'requires', r'supports',
        r'capable', r'limit\b',
    ]),
]


def classify_domain(text):
    """Classify domain by keyword matching. First match wins."""
    for domain, patterns in DOMAIN_RULES:
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return domain
    return "research"  # default


def classify_type(text):
    """Classify memory_type by keyword matching. First match wins."""
    for mtype, patterns in TYPE_RULES:
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return mtype
    return "state"  # default


def classify(text):
    """Classify a claim segment. Returns (domain, memory_type)."""
    return classify_domain(text), classify_type(text)


if __name__ == "__main__":
    import sys
    tests = [
        "GPT-2 achieves 135.9 tok/s at 37 dispatches on ANE",
        "The user never sees Subconscious",
        "ANE has a 93us dispatch floor on M5 Pro",
        "ane-compiler shipped on GitHub MIT",
        "ANE DMA at 111 GB/s",
        "Subconscious memory lifecycle has four phases",
        "Kill test passes 15/15 tokens against PyTorch",
    ]
    for t in tests:
        domain, mtype = classify(t)
        print(f"  [{domain:>10}] [{mtype:>12}] {t}")
