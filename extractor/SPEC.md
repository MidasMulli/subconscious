# Subconscious Extractor v0

After each 70B response, `extractor/` processes the conversation turn and writes memories to the store.

## Pipeline

```
conversation turn (text)
  -> claim_splitter.py    (regex: sentences, lists, colon-data)
  -> rule_classifier.py   (keywords: domain + memory_type)
  -> assembler.py          (entities: regex known-list + units)
  -> structured memories   (JSON: content, type, entities, domain, confidence, timestamp)
```

## Performance

- CPU only. No model in the hot path.
- Sub-second for any conversation turn.
- Zero poison by construction (content sourced from input text).

## Phase 0 Results (2026-04-01)

| Chunk | Gold | Extracted | Precision | Recall | Poison |
|-------|------|-----------|-----------|--------|--------|
| A1 (hardware) | 16 | 15 | 86.7% | 68.8% | 0.0% |
| B1 (production) | 15 | 16 | 81.2% | 60.0% | 0.0% |

## Known Limitations

- Conceptual/architectural text (C1-type) needs 70B idle-loop extraction, not this pipeline.
- Entity extraction is regex-only; add to KNOWN_ENTITIES list as new terms emerge.
- Recall gap is partly keyword matcher strictness, not splitter miss.

## Next Steps

- Wire into Midas agent post-response hook
- Add SQLite + ChromaDB storage backend
- Add 70B idle-loop extraction for conceptual memories
