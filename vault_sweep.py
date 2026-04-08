"""Vault sweep loop (Main 25 housekeeping deliverable, loop 9).

Scans the vault and adjacent project trees for completed deliverables that
no `vault/knowledge/*.md` file or `CLAUDE.md` entry references — i.e. work
that's already been done and shipped to disk but is structurally invisible
to the next session that comes looking for it.

This is the recurring lesson from Main 24 + Main 25:

  • Main 24: MiniLM CoreML compiled + validated 2026-03-25, sitting unused
    in `~/models/minilm-coreml/minilm.mlpackage`, never wired into daemon.
  • Main 24: O(n²) tokenizer.decode bug "diagnosed" but actually fixed in
    Main 21, the directive's diagnosis was 2 sessions stale.
  • Main 25 R1: ANE tile parallelism characterization complete in
    `vault/agent_reports/sram_scaling.md` + `ane_isa_catalog.md` +
    `concurrent_dispatch.md`, but no knowledge file references it.
  • Main 25 R2: ANE MMIO register survey complete in
    `vault/slc-probe/amcc_register_analysis.md` + `vault/agent_reports/iokit_selectors.md`,
    but project memory still uses superseded sel=2/sel=3 selector numbers.
  • Main 25 R3: Zin compiler pass catalog (dtrace-verified) complete in
    `vault/agent_reports/aned_internals.md`, no knowledge file references.
  • Main 25 R5: 8B GQA C+vDSP kernel built + benchmarked in
    `ane-compiler/llama_cpu_ops.c` + `bench_cpu_attention.py`, never wired
    into the production extractor `ane_extractor_8b.py`.

Five times across two sessions agents stopped because the work was
already done. This loop surfaces that pattern as a maintenance signal so
the next session sees the unwired deliverables instead of redoing them.

How it works:
  1. Walk fixed roots: `vault/agent_reports/`, `vault/ane-reverse/`,
     `~/models/`, `cowork/ngram-engine/` (top-level only), and
     `cowork/ane-compiler/` for `*.dylib` / `.mlpackage` artifacts.
  2. For each candidate file, build a stable reference token: filename
     stem (without extension), and the file's modification timestamp.
  3. Read every `vault/knowledge/*.md` and `CLAUDE.md` and `vault/CLAUDE_reference.md`.
     Concatenate. Lowercase. This is the "indexed corpus".
  4. A candidate is FOUND if its filename stem (case-insensitive) appears
     anywhere in the indexed corpus. Otherwise it's an UNREFERENCED orphan.
  5. Filter out trivially noisy stems (single-letter, all-digit, common
     words, file extension dupes).
  6. Sort by modification time descending — newest unreferenced artifact
     first, since recent work is most likely to be load-bearing.
  7. Write the report to `vault/memory/insights/unreferenced_artifacts-YYYYMMDD.md`
     so it appears in `MemoryBridge.get_insights()`.

Run this from `maintenance.run_all()` or as a standalone weekly cron.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger("vault_sweep")
if not log.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

VAULT = Path(os.path.expanduser("~/Desktop/cowork/vault"))
HOME = Path(os.path.expanduser("~"))
COWORK = Path(os.path.expanduser("~/Desktop/cowork"))

# Roots to scan for candidate deliverables
SCAN_ROOTS: list[tuple[Path, tuple[str, ...], int]] = [
    # (root, allowed extensions, max depth)
    (VAULT / "agent_reports", (".md",), 2),
    (VAULT / "ane-reverse", (".md",), 3),
    (HOME / "models", (".mlpackage", ".onnx", ".safetensors", ".bin"), 3),
    # ngram-engine top-level: only binaries. The directory has 150+ loose
    # exploratory .py probes that aren't shippable deliverables and would
    # flood the report with noise.
    (COWORK / "ngram-engine", (".dylib", ".so"), 1),
    (COWORK / "ane-compiler", (".dylib", ".so", ".mlpackage"), 1),
]

# Knowledge corpus to search for references
INDEX_FILES: list[Path] = [
    COWORK / "CLAUDE.md",
    VAULT / "CLAUDE_reference.md",
    VAULT / "CLAUDE_session_log.md",
] + sorted((VAULT / "knowledge").glob("*.md")) if (VAULT / "knowledge").exists() else [
    COWORK / "CLAUDE.md",
    VAULT / "CLAUDE_reference.md",
    VAULT / "CLAUDE_session_log.md",
]

# Stems we never want to flag (too generic, file-system noise)
STEM_BLACKLIST = {
    "readme", "init", "main", "test", "tests", "data", "input", "output",
    "tmp", "temp", "log", "config", "setup", "build", "run", "demo",
    "example", "examples", "util", "utils", "helper", "helpers", "common",
    "make", "all", "default", "final", "old", "new", "v1", "v2", "v3",
}

# Auto-generated session-bullet stems (canonical_inject pipeline / meta_memory_inject).
# These look orphaned because knowledge files don't reference them by stem,
# but they're correctly tracked by the meta-memory pipeline. Skip them.
SESSION_BULLET_PREFIX_RE = re.compile(
    r"^(main|build|track|session|day|week|phase)-\d+",
    re.IGNORECASE,
)
SESSION_BULLET_SUBSTRINGS = (
    "-complete-", "-pass-", "-fail-", "-kill-",
    "-decoded-", "-confirmed-", "-measured-",
)

# Binary deliverables — these are the real prizes, weight them
BINARY_EXTS = (".dylib", ".so", ".mlpackage", ".bin", ".npy",
                ".pt", ".safetensors", ".onnx")


def _is_binary_artifact(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(e) for e in BINARY_EXTS)


def _is_session_bullet(stem: str) -> bool:
    if SESSION_BULLET_PREFIX_RE.match(stem):
        return True
    if any(sub in stem for sub in SESSION_BULLET_SUBSTRINGS):
        return True
    return False


def _stem(path: Path) -> str:
    """Filename stem suitable for substring search."""
    name = path.name
    # Strip extension(s) — handle .mlpackage, .tar.gz, etc.
    while True:
        stem, ext = os.path.splitext(name)
        if not ext:
            break
        name = stem
    return name.lower()


def _walk(root: Path, exts: tuple[str, ...], max_depth: int):
    if not root.exists():
        return
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth > max_depth:
            dirnames.clear()
            continue
        # Skip hidden, __pycache__
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                       and d != "__pycache__"]
        for fn in filenames:
            if any(fn.endswith(e) for e in exts):
                yield Path(dirpath) / fn
        # mlpackage / mlmodelc are directories — emit them too
        for d in list(dirnames):
            if any(d.endswith(e) for e in exts):
                yield Path(dirpath) / d
                dirnames.remove(d)  # don't descend


def build_index_corpus() -> str:
    """Lowercase concatenation of every knowledge / dashboard / reference file."""
    parts = []
    for f in INDEX_FILES:
        if not f.exists():
            continue
        try:
            parts.append(f.read_text(errors="replace").lower())
        except Exception as e:
            log.warning(f"failed to read {f}: {e}")
    return "\n".join(parts)


def is_referenced(stem: str, corpus: str) -> bool:
    """Lowercase substring match. Tight enough to skip false positives via
    blacklist + min length."""
    if len(stem) < 4:
        return True
    if stem in STEM_BLACKLIST:
        return True
    if _is_session_bullet(stem):
        return True
    # Match either exact stem, or stem with - / _ swap
    s1 = stem
    s2 = stem.replace("_", "-")
    s3 = stem.replace("-", "_")
    return s1 in corpus or s2 in corpus or s3 in corpus


def vault_sweep(write_report: bool = True) -> dict:
    """Walk SCAN_ROOTS, find unreferenced artifacts. Returns stats dict."""
    corpus = build_index_corpus()
    log.info(f"index corpus: {len(corpus):,} chars across {len(INDEX_FILES)} files")

    candidates: list[Path] = []
    for root, exts, max_depth in SCAN_ROOTS:
        for path in _walk(root, exts, max_depth):
            candidates.append(path)
    log.info(f"scanned {len(candidates)} candidate artifacts across "
             f"{len(SCAN_ROOTS)} roots")

    unreferenced: list[tuple[Path, float, str, bool]] = []
    referenced_count = 0
    ane_reverse_dir = str(VAULT / "ane-reverse")
    for path in candidates:
        stem = _stem(path)
        if is_referenced(stem, corpus):
            referenced_count += 1
            continue
        # Auto-generated session-bullet reflections in vault/ane-reverse/
        # have long hyphenated names from canonical_inject. Skip them —
        # they're tracked by meta_memory_inject, not real orphans.
        if str(path).startswith(ane_reverse_dir) and stem.count("-") >= 4:
            referenced_count += 1
            continue
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0
        unreferenced.append((path, mtime, stem, _is_binary_artifact(path)))

    # Binary artifacts first (real prizes), then by mtime desc
    unreferenced.sort(key=lambda r: (not r[3], -r[1]))
    log.info(f"unreferenced artifacts: {len(unreferenced)} "
             f"(referenced: {referenced_count})")

    # Report
    report_path = None
    if write_report and unreferenced:
        out_dir = VAULT / "memory" / "insights"
        out_dir.mkdir(parents=True, exist_ok=True)
        date = time.strftime("%Y%m%d")
        report_path = out_dir / f"unreferenced_artifacts-{date}.md"
        with open(report_path, "w") as f:
            f.write(f"# Unreferenced artifacts sweep — {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"Scanned {len(candidates)} artifacts across {len(SCAN_ROOTS)} roots.\n")
            f.write(f"Referenced in CLAUDE.md / vault/knowledge/*: **{referenced_count}**\n")
            f.write(f"Unreferenced: **{len(unreferenced)}**\n\n")
            f.write("## Unreferenced (newest first)\n\n")
            f.write("| kind | stem | path | mtime |\n|---|---|---|---|\n")
            for path, mtime, stem, is_bin in unreferenced[:50]:
                rel = str(path).replace(str(HOME), "~")
                ts = time.strftime("%Y-%m-%d", time.localtime(mtime)) if mtime else "?"
                kind = "BIN" if is_bin else "doc"
                f.write(f"| {kind} | `{stem}` | `{rel}` | {ts} |\n")
            f.write("\n## Action\n\n")
            f.write("Each entry is a candidate for one of:\n")
            f.write("1. **Wire it up** — built but not referenced from production code\n")
            f.write("2. **Index it** — add a reference from a `vault/knowledge/*.md` file\n")
            f.write("3. **Archive it** — move to `vault/archive/` if no longer relevant\n")
            f.write("\nRun `python3 vault/subconscious/vault_sweep.py` to refresh.\n")
        log.info(f"report written: {report_path}")

    return {
        "scanned": len(candidates),
        "referenced": referenced_count,
        "unreferenced": len(unreferenced),
        "report_path": str(report_path) if report_path else None,
        "top_unreferenced": [
            {"stem": s, "path": str(p), "mtime": m, "binary": b}
            for p, m, s, b in unreferenced[:10]
        ],
    }


if __name__ == "__main__":
    import json
    result = vault_sweep()
    print(json.dumps(result, indent=2))
