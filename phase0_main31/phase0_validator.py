#!/usr/bin/env python3
"""Phase 0 validator — Main 31.

Score an extraction JSON against a ground-truth JSON for one of the three
schemas (Hardware Measurement, Dead Path, Architectural Finding).

Metrics:
  precision = correct_fields / total_extracted_fields  (excluding nulls)
  recall    = correct_fields / total_ground_truth_fields
  poison    = hallucinated_fields / total_extracted_fields
  malformed = parsing failures / total_records

Numeric tolerance: 5% for value fields.
String comparison: lowercase + strip + collapse whitespace.
Per-field and per-schema breakdown.

Usage:
    phase0_validator.py --schema 1 --extracted ext.json --truth gt.json
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Field categorization per schema ────────────────────────────────────────
SCHEMA_FIELDS = {
    1: {  # hardware measurement
        "key_fields": ["component", "property", "value", "unit"],
        "context_fields": ["measurement_method", "session", "confidence", "supersedes"],
        # For schema 1, match records by (component, value, unit) — these are
        # the natural key. Property name is a free-text label that varies
        # between extractor and annotator; we score it as a context field.
        "match_func": lambda a, b: (
            normalize(a.get("component")) == normalize(b.get("component"))
            and a.get("unit") == b.get("unit")
            and numeric_close(a.get("value"), b.get("value"))
        ),
        "id_func": lambda r: f'{r.get("component", "?")}::{r.get("value")}{r.get("unit", "")}',
        "numeric_fields": {"value"},
    },
    2: {  # dead path
        "key_fields": ["path_name", "status", "evidence_session"],
        "context_fields": ["original_hypothesis", "evidence_against", "scope", "remaining_leads"],
        # Match by fuzzy path_name (substring overlap)
        "match_func": lambda a, b: (
            normalize(a.get("path_name", "")) == normalize(b.get("path_name", ""))
            or normalize(a.get("path_name", ""))[:30] in normalize(b.get("path_name", ""))
            or normalize(b.get("path_name", ""))[:30] in normalize(a.get("path_name", ""))
        ),
        "id_func": lambda r: normalize(r.get("path_name", "?")),
        "numeric_fields": set(),
    },
    3: {  # architectural finding
        "key_fields": ["finding", "status"],
        "context_fields": ["detail", "evidence_chain", "implications", "contradicts"],
        # Match by fuzzy finding overlap on first 60 chars
        "match_func": lambda a, b: (
            normalize(a.get("finding", ""))[:60] == normalize(b.get("finding", ""))[:60]
            or normalize(a.get("finding", ""))[:30] in normalize(b.get("finding", ""))
            or normalize(b.get("finding", ""))[:30] in normalize(a.get("finding", ""))
        ),
        "id_func": lambda r: normalize(r.get("finding", "?"))[:80],
        "numeric_fields": set(),
    },
}

NUMERIC_TOLERANCE = 0.05  # 5%


def normalize(s):
    if s is None:
        return ""
    return re.sub(r'\s+', ' ', str(s).lower().strip())


def numeric_close(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(abs(a), abs(b)) <= NUMERIC_TOLERANCE


def field_matches(extracted_val, truth_val, is_numeric):
    if extracted_val is None and truth_val is None:
        return True
    if extracted_val is None or truth_val is None:
        return False
    if is_numeric:
        return numeric_close(extracted_val, truth_val)
    if isinstance(extracted_val, list) and isinstance(truth_val, list):
        # List equality: at least one match per truth item
        if not truth_val and not extracted_val:
            return True
        ne = [normalize(x) for x in extracted_val]
        nt = [normalize(x) for x in truth_val]
        return any(t in ne or any(t in e or e in t for e in ne) for t in nt)
    if isinstance(extracted_val, dict) and isinstance(truth_val, dict):
        # For evidence_chain entries, compare on session field
        return normalize(extracted_val.get("session")) == normalize(truth_val.get("session"))
    return normalize(extracted_val) == normalize(truth_val)


def score(extracted, truth, schema_id):
    cfg = SCHEMA_FIELDS[schema_id]
    id_func = cfg["id_func"]
    match_func = cfg["match_func"]
    numeric_fields = cfg["numeric_fields"]
    all_fields = cfg["key_fields"] + cfg["context_fields"]

    field_scores = defaultdict(lambda: {"correct": 0, "wrong": 0, "missing": 0, "extra": 0})
    record_status = []

    # Bipartite matching: greedy match each truth record to the best extracted candidate
    matched_truth_idx = set()
    matched_ext_idx = set()
    pairs = []
    for ti, t_rec in enumerate(truth):
        for ei, e_rec in enumerate(extracted):
            if ei in matched_ext_idx:
                continue
            try:
                if match_func(e_rec, t_rec):
                    pairs.append((ti, ei))
                    matched_truth_idx.add(ti)
                    matched_ext_idx.add(ei)
                    break
            except (TypeError, KeyError):
                continue

    only_truth = [i for i in range(len(truth)) if i not in matched_truth_idx]
    only_ext = [i for i in range(len(extracted)) if i not in matched_ext_idx]

    for ti, ei in pairs:
        t_rec = truth[ti]
        e_rec = extracted[ei]
        rec_errors = []
        for f in all_fields:
            e_val = e_rec.get(f)
            t_val = t_rec.get(f)
            is_num = f in numeric_fields
            if field_matches(e_val, t_val, is_num):
                field_scores[f]["correct"] += 1
            else:
                field_scores[f]["wrong"] += 1
                rec_errors.append(f"{f}: ext={e_val!r} truth={t_val!r}")
        record_status.append({"id": id_func(t_rec)[:60], "matched": True, "errors": rec_errors})

    for ti in only_truth:
        record_status.append({"id": id_func(truth[ti])[:60], "matched": False, "missed": True})
        for f in all_fields:
            if truth[ti].get(f) is not None:
                field_scores[f]["missing"] += 1

    for ei in only_ext:
        record_status.append({"id": id_func(extracted[ei])[:60], "matched": False, "hallucinated": True})
        for f in all_fields:
            if extracted[ei].get(f) is not None:
                field_scores[f]["extra"] += 1

    matched_ids = pairs
    only_extracted = only_ext

    # Aggregate
    total_correct = sum(s["correct"] for s in field_scores.values())
    total_wrong = sum(s["wrong"] for s in field_scores.values())
    total_missing = sum(s["missing"] for s in field_scores.values())
    total_extra = sum(s["extra"] for s in field_scores.values())

    total_extracted_field_values = total_correct + total_wrong + total_extra
    total_truth_field_values = total_correct + total_wrong + total_missing

    precision = total_correct / total_extracted_field_values if total_extracted_field_values else 0.0
    recall = total_correct / total_truth_field_values if total_truth_field_values else 0.0
    poison = (total_extra + total_wrong) / total_extracted_field_values if total_extracted_field_values else 0.0

    return {
        "schema_id": schema_id,
        "n_extracted": len(extracted),
        "n_truth": len(truth),
        "n_matched": len(pairs),
        "n_missed": len(only_truth),
        "n_hallucinated": len(only_ext),
        "field_scores": dict(field_scores),
        "totals": {
            "correct": total_correct,
            "wrong": total_wrong,
            "missing": total_missing,
            "extra": total_extra,
        },
        "metrics": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "poison": round(poison, 3),
        },
        "record_status": record_status,
    }


def gate(metrics):
    p, r, po = metrics["precision"], metrics["recall"], metrics["poison"]
    if p > 0.90 and r > 0.70 and po < 0.05:
        return "PASS"
    fails = []
    if p <= 0.90: fails.append(f"precision={p}")
    if r <= 0.70: fails.append(f"recall={r}")
    if po >= 0.05: fails.append(f"poison={po}")
    if len(fails) == 1 and po < 0.10:
        return f"CONDITIONAL ({fails[0]} just below threshold)"
    if po >= 0.10:
        return f"FAIL (poison={po} >= 10%)"
    return f"FAIL ({', '.join(fails)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--extracted", required=True, help="Path to extracted JSON array")
    ap.add_argument("--truth", required=True, help="Path to ground truth JSON array")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        extracted = json.loads(Path(args.extracted).read_text())
    except json.JSONDecodeError as e:
        print(f"MALFORMED extracted JSON: {e}")
        sys.exit(2)
    try:
        truth_blob = json.loads(Path(args.truth).read_text())
        # Truth files store all 3 schemas in one envelope; pick the requested one
        if isinstance(truth_blob, dict):
            key = {1: "schema_1_hardware_measurement",
                   2: "schema_2_dead_path",
                   3: "schema_3_architectural_finding"}[args.schema]
            truth = truth_blob.get(key, [])
        else:
            truth = truth_blob
    except json.JSONDecodeError as e:
        print(f"MALFORMED truth JSON: {e}")
        sys.exit(2)

    result = score(extracted, truth, args.schema)
    print(f"=== Schema {args.schema} extraction scorecard ===")
    print(f"  Extracted records: {result['n_extracted']}")
    print(f"  Truth records:     {result['n_truth']}")
    print(f"  Matched:           {result['n_matched']}")
    print(f"  Missed (recall loss):     {result['n_missed']}")
    print(f"  Hallucinated (poison):    {result['n_hallucinated']}")
    print()
    print(f"  Precision: {result['metrics']['precision']}")
    print(f"  Recall:    {result['metrics']['recall']}")
    print(f"  Poison:    {result['metrics']['poison']}")
    print()
    print(f"  GATE: {gate(result['metrics'])}")
    print()
    print("=== Per-field breakdown ===")
    for f, s in sorted(result["field_scores"].items()):
        print(f"  {f:25s}  correct={s['correct']:3}  wrong={s['wrong']:3}  missing={s['missing']:3}  extra={s['extra']:3}")

    if args.verbose:
        print()
        print("=== Record-level errors ===")
        for r in result["record_status"]:
            if r.get("errors"):
                print(f"  [{r['id'][:50]}]")
                for e in r["errors"][:3]:
                    print(f"    - {e}")
            elif r.get("missed"):
                print(f"  [MISSED] {r['id'][:50]}")
            elif r.get("hallucinated"):
                print(f"  [HALLUCINATED] {r['id'][:50]}")


if __name__ == "__main__":
    main()
