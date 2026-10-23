"""
Fusion Selection Protocol — Removing Test-Set Oracle Selection
=============================================================

Reviewer auZN, marked critical to their acceptance recommendation:

    "Table 4 reports the better of Simple Fusion and Contextual Phrasing
    separately for each model-operator cell. The authors should report both
    formulations separately and use a single deployable selection policy for
    the primary results. Acceptable policies include choosing one formulation
    globally, selecting it on a disjoint validation set, or averaging the two.
    The per-cell maximum may be retained only as an explicitly labeled oracle
    upper bound."

This is correct: picking the better fusion per cell using test-set outcomes is
oracle selection and inflates reported gains. This script implements the fix.

Policies implemented
--------------------
  val        Select the fusion realization on the VAL split, per template,
             then apply that choice to TEST. Most defensible.
  val-global Select ONE fusion realization on VAL by overall mAP, apply
             everywhere on TEST. Simplest to state.
  average    Report the mean of the two realizations on TEST.
  oracle     Per-cell maximum on TEST. Reported only as a labelled upper bound.

Inputs
------
For each encoder you need four metrics JSON files produced by evaluate.py:
    val  + SIMPLE      val  + QUEST
    test + SIMPLE      test + QUEST

Obtain them by running retrieve_fuzzy.py with --augment_type=SIMPLE and
--augment_type=QUEST on both splits, then evaluate.py on each result.

Usage
-----
    python fusion_selection.py \
        --val_simple  debug_out/retrieval/results.decompose1_BGE-val-simple.metrics.json \
        --val_quest   debug_out/retrieval/results.decompose1_BGE-val-quest.metrics.json \
        --test_simple debug_out/retrieval/results.decompose1_BGE-test-simple.metrics.json \
        --test_quest  debug_out/retrieval/results.decompose1_BGE-test-quest.metrics.json \
        --name BGE \
        --output debug_out/analysis/fusion_BGE.json
"""

import os
import json
import argparse
import numpy as np


# Canonical template order for reporting
TEMPLATE_ORDER = [
    "_ that are also _",
    "_ that are also both _ and _",
    "_ that are not _",
    "_ that are also _ but not _",
    "_ or _",
    "_ or _ or _",
]

TEMPLATE_LABEL = {
    "_ that are also _": "A AND B",
    "_ that are also both _ and _": "A AND B AND C",
    "_ that are not _": "A AND NOT B",
    "_ that are also _ but not _": "A AND B AND NOT C",
    "_ or _": "A OR B",
    "_ or _ or _": "A OR B OR C",
}


def extract_map(metrics_path):
    """
    Extract per-template mAP@100 from an evaluate.py metrics JSON.

    Returns {template_string: map_value}. Keys in the JSON look like
    "<template>_pos"; the trailing split marker is stripped.
    """
    with open(metrics_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    out = {}
    for key, metrics in data.items():
        if key.startswith("_"):  # skip _statistical_significance etc.
            if key == "_statistical_significance":
                continue
        if not isinstance(metrics, dict):
            continue

        template = key[:-4] if key.endswith("_pos") else key

        # evaluate.py nests mAP under a 'map@k' dict keyed by cutoff
        map_block = metrics.get("map@k")
        if not isinstance(map_block, dict) or not map_block:
            continue

        # Prefer cutoff 100 when present, else the largest available cutoff
        if 100 in map_block:
            value = map_block[100]
        elif "100" in map_block:
            value = map_block["100"]
        else:
            cutoff = sorted(map_block.keys(), key=lambda c: int(c))[-1]
            value = map_block[cutoff]

        out[template] = float(value)

    return out


def apply_policy(policy, val_simple, val_quest, test_simple, test_quest):
    """
    Return (selected_scores, choices) under the given policy.

    selected_scores maps template -> mAP on TEST.
    choices maps template -> 'SIMPLE' | 'QUEST' | 'average' | 'oracle:<x>'.
    """
    templates = [t for t in TEMPLATE_ORDER if t in test_simple and t in test_quest]
    scores, choices = {}, {}

    if policy == "val":
        for t in templates:
            if t not in val_simple or t not in val_quest:
                # No val evidence — fall back to the global val winner
                pick = "QUEST" if _global_winner(val_simple, val_quest) == "QUEST" else "SIMPLE"
            else:
                pick = "SIMPLE" if val_simple[t] >= val_quest[t] else "QUEST"
            scores[t] = test_simple[t] if pick == "SIMPLE" else test_quest[t]
            choices[t] = pick

    elif policy == "val-global":
        pick = _global_winner(val_simple, val_quest)
        for t in templates:
            scores[t] = test_simple[t] if pick == "SIMPLE" else test_quest[t]
            choices[t] = pick

    elif policy == "average":
        for t in templates:
            scores[t] = 0.5 * (test_simple[t] + test_quest[t])
            choices[t] = "average"

    elif policy == "oracle":
        for t in templates:
            if test_simple[t] >= test_quest[t]:
                scores[t] = test_simple[t]
                choices[t] = "oracle:SIMPLE"
            else:
                scores[t] = test_quest[t]
                choices[t] = "oracle:QUEST"

    else:
        raise ValueError(f"unknown policy: {policy}")

    return scores, choices


def _global_winner(val_simple, val_quest):
    """Return whichever realization has the higher mean mAP on val."""
    common = set(val_simple) & set(val_quest)
    if not common:
        return "QUEST"
    mean_simple = np.mean([val_simple[t] for t in common])
    mean_quest = np.mean([val_quest[t] for t in common])
    return "SIMPLE" if mean_simple >= mean_quest else "QUEST"


def main():
    parser = argparse.ArgumentParser(
        description="Apply a deployable fusion-selection policy and remove oracle selection"
    )
    parser.add_argument("--val_simple", required=True)
    parser.add_argument("--val_quest", required=True)
    parser.add_argument("--test_simple", required=True)
    parser.add_argument("--test_quest", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    val_simple = extract_map(args.val_simple)
    val_quest = extract_map(args.val_quest)
    test_simple = extract_map(args.test_simple)
    test_quest = extract_map(args.test_quest)

    templates = [t for t in TEMPLATE_ORDER if t in test_simple and t in test_quest]
    if not templates:
        print("No overlapping templates found. Check the metrics JSON structure.")
        return

    policies = ["val", "val-global", "average", "oracle"]
    results = {}
    for policy in policies:
        scores, choices = apply_policy(
            policy, val_simple, val_quest, test_simple, test_quest
        )
        results[policy] = {
            "per_template": {t: scores[t] for t in templates},
            "choices": {t: choices[t] for t in templates},
            "average": float(np.mean([scores[t] for t in templates])),
        }

    # ---- Report ----
    print("\n" + "=" * 78)
    print(f"Fusion Selection Protocol — {args.name}")
    print("=" * 78)

    print("\nRaw per-realization mAP@100 on TEST:")
    header = f"  {'Template':<22}{'SIMPLE':>10}{'QUEST':>10}"
    print(header)
    for t in templates:
        print(f"  {TEMPLATE_LABEL.get(t, t):<22}"
              f"{test_simple[t]:>10.4f}{test_quest[t]:>10.4f}")
    print(f"  {'AVERAGE':<22}"
          f"{np.mean([test_simple[t] for t in templates]):>10.4f}"
          f"{np.mean([test_quest[t] for t in templates]):>10.4f}")

    print("\nmAP@100 on TEST under each selection policy:")
    print(f"  {'Template':<22}" + "".join(f"{p:>13}" for p in policies))
    for t in templates:
        row = f"  {TEMPLATE_LABEL.get(t, t):<22}"
        for p in policies:
            row += f"{results[p]['per_template'][t]:>13.4f}"
        print(row)
    row = f"  {'AVERAGE':<22}"
    for p in policies:
        row += f"{results[p]['average']:>13.4f}"
    print(row)

    print("\nFusion chosen per template under the 'val' policy:")
    for t in templates:
        print(f"  {TEMPLATE_LABEL.get(t, t):<22}{results['val']['choices'][t]}")
    print(f"\n  'val-global' selected: {results['val-global']['choices'][templates[0]]}")

    inflation = results["oracle"]["average"] - results["val"]["average"]
    print("\n" + "-" * 78)
    print(f"Oracle inflation (oracle average minus val-selected average): {inflation:+.4f}")
    if inflation > 0.005:
        print("This is the magnitude by which per-cell test-set selection inflated")
        print("the originally reported numbers. Report the val-selected figures as")
        print("primary and the oracle figures only as a labelled upper bound.")
    else:
        print("Inflation is small. The original conclusions likely survive the")
        print("protocol change, which is a helpful thing to be able to state.")
    print("-" * 78)

    print("\nRecommended: adopt the 'val' policy for the primary results table,")
    print("report SIMPLE and QUEST columns separately in an appendix, and label")
    print("the oracle row explicitly as an upper bound. Then recompute averages,")
    print("relative improvements, significance tests, abstract, and conclusions")
    print("from the val-selected numbers only.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        payload = {
            "name": args.name,
            "test_simple": test_simple,
            "test_quest": test_quest,
            "val_simple": val_simple,
            "val_quest": val_quest,
            "policies": results,
            "oracle_inflation": float(inflation),
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
