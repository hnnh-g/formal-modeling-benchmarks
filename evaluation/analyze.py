import json
import argparse
from collections import defaultdict


def load_modeling_results(filename, domain_filter=None):
    """
    Loads the JSONL results file produced by evaluate.py.
    Returns the list of valid samples (skipping any where the judge failed).
    Simple and focused — no need to separate into tiers at load time,
    since we compute all breakdowns from the same list afterwards.
    """
    samples = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)

            # Skip samples where GPT-4o-mini couldn't produce a valid score
            if sample.get("score") == -1:
                continue

            # Apply domain filter if requested
            if domain_filter and sample.get("domain") != domain_filter:
                continue

            samples.append(sample)

    return samples

def map_to_rating(accuracy_pct):
    """Maps 0-100% accuracy to your 1-5 model selection rating scale."""
    if accuracy_pct >= 90: return 5
    elif accuracy_pct >= 80: return 4
    elif accuracy_pct >= 70: return 3
    elif accuracy_pct >= 60: return 2
    else: return 1


def compute_results(samples):
    if not samples:
        print("No valid samples found.")
        return

    # ── Overall accuracy ──────────────────────────────────────────────
    total_score = sum(s["score"] for s in samples)
    max_score   = len(samples) * 2
    overall_acc = (total_score / max_score) * 100

    print("\n" + "=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)
    print(f"  Questions evaluated : {len(samples)}")
    print(f"  Total score         : {total_score} / {max_score}")
    print(f"  Accuracy            : {overall_acc:.1f}%")
    print(f"  Rating              : {map_to_rating(overall_acc)}/5")

    # ── Per-domain breakdown ──────────────────────────────────────────
    domain_stats = defaultdict(lambda: {"score_sum": 0, "count": 0})
    for s in samples:
        domain_stats[s["domain"]]["score_sum"] += s["score"]
        domain_stats[s["domain"]]["count"]     += 1

    print("\n" + "=" * 50)
    print("PER-DOMAIN BREAKDOWN")
    print("=" * 50)
    for domain, stats in sorted(domain_stats.items()):
        domain_max = stats["count"] * 2
        domain_acc = (stats["score_sum"] / domain_max) * 100
        print(f"\n  Domain   : {domain}")
        print(f"  Score    : {stats['score_sum']} / {domain_max}")
        print(f"  Accuracy : {domain_acc:.1f}%")
        print(f"  Rating   : {map_to_rating(domain_acc)}/5")

    # ── Brief failure summary (counts only, no full text) ─────────────
    # Keeping just the counts gives you a quick signal of how many
    # questions need attention without printing walls of model output.
    failed  = sum(1 for s in samples if s["score"] == 0)
    partial = sum(1 for s in samples if s["score"] == 1)

    print("\n" + "=" * 50)
    print("FAILURE SUMMARY")
    print("=" * 50)
    print(f"  Score 0 (failed)  : {failed} questions")
    print(f"  Score 1 (partial) : {partial} questions")
    print(f"  Score 2 (correct) : {len(samples) - failed - partial} questions")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute accuracy results from modeling benchmark evaluation"
    )
    parser.add_argument("--result", required=True,
                        help="Path to the JSONL results file from evaluate.py")
    parser.add_argument("--domain", default=None,
                        choices=["UML", "BPMN", "Petri Nets"],
                        help="Filter to one domain (default: all domains)")
    args = parser.parse_args()

    samples = load_modeling_results(args.result, domain_filter=args.domain)
    compute_results(samples)
