import json
from collections import Counter, defaultdict


def summarize_evaluations(filepath="evaluated_payload_results.jsonl"):
    print(f"[*] Loading evaluations from {filepath}...\n")

    total_evaluated = 0
    successful_evals = 0
    failed_evals = 0

    # Track anomalous vs helpful
    anomalous_count = 0
    helpful_count = 0

    # Breakdowns
    categories = Counter()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            total_evaluated += 1

            if data.get("eval_status") != "success":
                failed_evals += 1
                continue

            successful_evals += 1

            # Tally anomalous vs helpful
            is_anomalous = data.get("eval_is_anomalous")
            if is_anomalous:
                anomalous_count += 1
            else:
                helpful_count += 1

            # Tally categories
            category = data.get("eval_category", "Unknown")
            categories[category] += 1

    # Print the Summary Report
    print("=" * 50)
    print("📊 PAYLOAD EXECUTION SUMMARY REPORT 📊")
    print("=" * 50)
    print(f"Total Records Processed: {total_evaluated}")
    print(f"Successful LLM Evals:    {successful_evals}")
    print(f"Failed/Blocked Evals:    {failed_evals}")
    print("-" * 50)

    if successful_evals == 0:
        print("[!] No successful evaluations to summarize.")
        return

    anomalous_pct = (anomalous_count / successful_evals) * 100
    helpful_pct = (helpful_count / successful_evals) * 100

    print(
        f"🚨 ANOMALOUS BEHAVIOR (Trigger Success/Partial): {anomalous_count} ({anomalous_pct:.1f}%)"
    )
    print(
        f"🛡️ HELPFUL BEHAVIOR (Trigger Failed/Ignored):   {helpful_count} ({helpful_pct:.1f}%)"
    )
    print("-" * 50)

    print("Categorical Breakdown:")
    for cat, count in categories.most_common():
        cat_pct = (count / successful_evals) * 100
        print(f"  - {cat:<25}: {count:>4} ({cat_pct:.1f}%)")

    print("=" * 50)


if __name__ == "__main__":
    summarize_evaluations(
        filepath="/app/evaluated_payload_results__live_fire_results_layer15_20_35_55_trigger_exec_isoforest_deep_sweep_02_20260401_044141.jsonl"
    )
