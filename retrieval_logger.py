#!/usr/bin/env python3
"""
Subconscious 1B: Retrieval quality logger.

Hooks into the agent to log every retrieval call with:
- Query text and embedding
- Top-10 results with similarity scores
- Which memories the 70B actually referenced in its response

After 10 conversations, analyze to calibrate threshold and K.

Usage: import and call log_retrieval() from agent_v2.py

Copyright 2026 Nick Lo. MIT License.
"""

import json
import os
import time
from datetime import datetime

LOG_DIR = os.path.expanduser("~/Desktop/cowork/vault/subconscious/retrieval_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def log_retrieval(query, results_top10, response_text, tool_name=None):
    """Log a single retrieval event for calibration analysis.

    Args:
        query: user's message text
        results_top10: list of dicts with 'text', 'score', 'type' (top 10)
        response_text: the 70B's final response
        tool_name: which tool was used (or 'conversation')
    """
    # Check which retrieved memories appear in the response
    referenced = []
    for i, r in enumerate(results_top10):
        mem_text = r.get('text', '')
        # Simple check: do significant words from the memory appear in the response?
        mem_words = set(w.lower().strip('.,;:()') for w in mem_text.split()
                       if len(w) > 5)
        resp_words = set(w.lower().strip('.,;:()') for w in response_text.split()
                        if len(w) > 5)
        overlap = len(mem_words & resp_words)
        total = len(mem_words) if mem_words else 1
        usage_score = overlap / total
        referenced.append({
            'rank': i + 1,
            'score': r.get('score', 0),
            'text_preview': mem_text[:100],
            'usage_score': round(usage_score, 3),
            'likely_used': usage_score > 0.3,
        })

    entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query[:200],
        'tool': tool_name,
        'n_results': len(results_top10),
        'results': referenced,
        'response_length': len(response_text),
        'n_likely_used': sum(1 for r in referenced if r['likely_used']),
    }

    # Append to daily log
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_path = os.path.join(LOG_DIR, f'retrieval_{date_str}.jsonl')
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    return entry


def analyze_logs():
    """Analyze retrieval logs for calibration.

    Reports:
    - How often is memory #6-10 more relevant than #1-5?
    - How often does the 70B ignore retrieved memories?
    - Similarity score distribution of useful vs ignored
    """
    all_entries = []
    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith('.jsonl'):
            continue
        with open(os.path.join(LOG_DIR, fname)) as f:
            for line in f:
                if line.strip():
                    all_entries.append(json.loads(line))

    if not all_entries:
        print("No retrieval logs found")
        return

    print(f"Analyzing {len(all_entries)} retrieval events\n")

    # Stats
    total_used = 0
    total_ignored = 0
    used_scores = []
    ignored_scores = []
    rank_6_10_useful = 0
    rank_1_5_ignored = 0

    for entry in all_entries:
        for r in entry['results']:
            if r['likely_used']:
                total_used += 1
                used_scores.append(r['score'])
                if r['rank'] > 5:
                    rank_6_10_useful += 1
            else:
                total_ignored += 1
                ignored_scores.append(r['score'])
                if r['rank'] <= 5:
                    rank_1_5_ignored += 1

    total = total_used + total_ignored
    print(f"Total memories retrieved: {total}")
    print(f"  Used by 70B: {total_used} ({total_used/max(1,total):.1%})")
    print(f"  Ignored: {total_ignored} ({total_ignored/max(1,total):.1%})")
    print(f"  Rank 6-10 useful: {rank_6_10_useful}")
    print(f"  Rank 1-5 ignored: {rank_1_5_ignored}")

    if used_scores:
        import numpy as np
        print(f"\nUsed memory scores: min={min(used_scores):.3f} "
              f"median={np.median(used_scores):.3f} max={max(used_scores):.3f}")
    if ignored_scores:
        import numpy as np
        print(f"Ignored memory scores: min={min(ignored_scores):.3f} "
              f"median={np.median(ignored_scores):.3f} max={max(ignored_scores):.3f}")

    # Recommendation
    if used_scores and ignored_scores:
        import numpy as np
        threshold_suggestion = np.percentile(used_scores, 25)
        print(f"\nSuggested threshold: {threshold_suggestion:.3f} "
              f"(25th percentile of used scores)")
        k_suggestion = 5 if rank_6_10_useful < 3 else 10
        print(f"Suggested K: {k_suggestion}")


if __name__ == "__main__":
    analyze_logs()
