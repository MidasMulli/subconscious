"""Shared embedder factory: prefer CoreML MiniLM (ANE), fall back to CPU.

Used by canonical_inject, semantic_supersede, conceptual_extractor, and
meta_memory_inject so all four maintenance loops embed via the same path.
"""
from __future__ import annotations

import os
import sys


def get_embedder():
    """Return a SentenceTransformer-compatible embedder.

    Tries the precompiled CoreML MiniLM at ~/models/minilm-coreml/ first
    (routes through ANE via CPU_AND_NE). Falls back to CPU
    sentence-transformers if missing or `MIDAS_DISABLE_COREML_EMBED=1`.
    """
    try:
        sys.path.insert(0, os.path.expanduser(
            "~/Desktop/cowork/orion-ane/memory"))
        from coreml_embedder import maybe_load_coreml_embedder
        coreml = maybe_load_coreml_embedder()
        if coreml is not None:
            return coreml
    except Exception as e:
        print(f"[_embedder] CoreML load failed: {e}; using CPU fallback",
              file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
