"""
Experiment A: Classical t-norm / t-conorm Baselines  (v2 — corrected)
=====================================================================

Changes from v1
---------------
1. FIXED: the mixed template "A and B and not C" was previously classified into
   a generic "not" family that negated every atom after the first, i.e. it
   computed min(S_A, 1-S_B, 1-S_C) — wrongly negating B. Templates now map to
   explicit (positive_atoms, negated_atoms) structures.
2. ADDED: a pool-normalized Godel variant ("godel-poolnorm") that min-max
   normalizes each atom's scores within the candidate pool before applying
   standard fuzzy negation. This isolates the calibration question: if a
   calibrated Godel still underperforms NSFL, the fused-score signal matters
   beyond calibration — and it preempts the objection that only strawman
   baselines were compared.

These baselines rerank the SAME candidate pool as NSFL but see only atomic
scores — no access to S_AB / S_ABC, the signal NSFL is built on. The gap
between them and NSFL isolates the Neuro-Symbolic Delta contribution.

Output format matches evaluate.py:  qid \t docid \t rank \t score

Usage
-----
    python baseline_tnorms.py \
        --score_dump debug_out/scores/scores_BGE.tsv \
        --out_dir debug_out/retrieval \
        --name BGE
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------
# Template structures: which atoms are positive, which are negated, and
# whether the connective is conjunctive or disjunctive.
# Update the strings here if this repo uses different template text.
# ----------------------------------------------------------------------------

TEMPLATE_STRUCTURE = {
    # template string                     connective  positive          negated
    "_ that are also _":                  ("and", ["S_A", "S_B"],        []),
    "_ that are also both _ and _":       ("and", ["S_A", "S_B", "S_C"], []),
    "_ that are not _":                   ("and", ["S_A"],               ["S_B"]),
    "_ that are also _ but not _":        ("and", ["S_A", "S_B"],        ["S_C"]),
    "_ or _":                             ("or",  ["S_A", "S_B"],        []),
    "_ or _ or _":                        ("or",  ["S_A", "S_B", "S_C"], []),
}


# ----------------------------------------------------------------------------
# Fuzzy families
# ----------------------------------------------------------------------------
# Each family combines a list of truth values (positives plus negated
# negatives) under its t-norm / t-conorm. Negation is standard: n(s) = 1 - s.


def godel_combine(values, connective):
    if connective == "and":
        return min(values)
    return max(values)


def product_combine(values, connective):
    if connective == "and":
        out = 1.0
        for v in values:
            out *= v
        return out
    # probabilistic sum, applied iteratively
    out = values[0]
    for v in values[1:]:
        out = out + v - out * v
    return out


def lukasiewicz_combine(values, connective):
    if connective == "and":
        return max(0.0, sum(values) - (len(values) - 1))
    return min(1.0, sum(values))


FAMILIES = {
    "godel": godel_combine,
    "product": product_combine,
    "lukasiewicz": lukasiewicz_combine,
}


def score_row(row, structure, combine_fn, negation=lambda s: 1.0 - s):
    """Score one candidate under one fuzzy family, honoring atom roles."""
    connective, positive, negated = structure

    values = []
    for key in positive:
        v = row[key]
        if np.isnan(v):
            return float("nan")
        values.append(v)
    for key in negated:
        v = row[key]
        if np.isnan(v):
            return float("nan")
        values.append(negation(v))

    if not values:
        return float("nan")
    return combine_fn(values, connective)


# ----------------------------------------------------------------------------
# Pool-normalized Godel ("calibrated" variant)
# ----------------------------------------------------------------------------

def poolnorm_columns(rows, keys=("S_A", "S_B", "S_C")):
    """
    Min-max normalize each atom column within a query's candidate pool.

    Returns new row dicts with normalized values. Columns entirely NaN are
    untouched; constant columns map to 0.5.
    """
    eps = 1e-8
    normed = [dict(r) for r in rows]
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        finite = vals[~np.isnan(vals)]
        if finite.size == 0:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < eps:
            for r in normed:
                if not np.isnan(r[key]):
                    r[key] = 0.5
            continue
        for r in normed:
            if not np.isnan(r[key]):
                r[key] = (r[key] - lo) / (hi - lo + eps)
    return normed


# ----------------------------------------------------------------------------
# Dump loading
# ----------------------------------------------------------------------------

def load_score_dump(path):
    """Load a score dump into {qid: {'template': str, 'rows': [dict, ...]}}."""
    queries = defaultdict(lambda: {"template": None, "rows": []})
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            def f(key):
                v = row.get(key, "NaN")
                return float("nan") if v in (None, "", "NaN") else float(v)

            qid = row["qid"]
            queries[qid]["template"] = row["template"]
            queries[qid]["rows"].append({
                "docid": row["docid"],
                "S_A": f("S_A"), "S_B": f("S_B"), "S_C": f("S_C"),
            })
    return queries


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_baselines(score_dump, out_dir, name, top_k=100):
    print(f"Loading score dump: {score_dump}")
    queries = load_score_dump(score_dump)
    print(f"  {len(queries)} queries loaded")

    unknown_templates = sorted({
        p["template"] for p in queries.values()
        if p["template"] not in TEMPLATE_STRUCTURE
    })
    if unknown_templates:
        print("  WARNING — templates not in TEMPLATE_STRUCTURE (will be skipped):")
        for t in unknown_templates:
            print(f"    {t!r}")
        print("  Add them to TEMPLATE_STRUCTURE with correct atom roles and rerun.")

    os.makedirs(out_dir, exist_ok=True)

    baseline_names = list(FAMILIES.keys()) + ["godel-poolnorm"]
    diagnostics = {}

    for baseline_name in baseline_names:
        out_path = os.path.join(out_dir, f"results.tnorm-{baseline_name}_{name}.txt")
        n_written = 0
        n_zero = 0
        n_total = 0

        with open(out_path, "w", encoding="utf-8") as out_fh:
            for qid, payload in queries.items():
                structure = TEMPLATE_STRUCTURE.get(payload["template"])
                if structure is None:
                    continue

                if baseline_name == "godel-poolnorm":
                    rows = poolnorm_columns(payload["rows"])
                    combine_fn = FAMILIES["godel"]
                else:
                    rows = payload["rows"]
                    combine_fn = FAMILIES[baseline_name]

                scored = []
                for r in rows:
                    s = score_row(r, structure, combine_fn)
                    if np.isnan(s):
                        continue
                    scored.append((r["docid"], float(s)))
                    n_total += 1
                    if abs(s) < 1e-12:
                        n_zero += 1

                scored.sort(key=lambda x: x[1], reverse=True)
                for rank, (docid, score) in enumerate(scored[:top_k], start=1):
                    out_fh.write(f"{qid}\t{docid}\t{rank}\t{score:.6f}\n")
                    n_written += 1

        frac_zero = (n_zero / n_total) if n_total else 0.0
        diagnostics[baseline_name] = frac_zero
        print(f"  [{baseline_name:16s}] wrote {n_written} rows -> {out_path}")

    print("\nDegenerate-score summary (fraction of scores exactly zero):")
    for baseline_name, frac in diagnostics.items():
        print(f"  {baseline_name:16s} {frac*100:6.2f}%")
    print(
        "\nHigh collapse for the raw Lukasiewicz family is expected on\n"
        "uncalibrated scores and is itself reportable. The godel-poolnorm\n"
        "variant is the fair 'calibrated' comparison: if it still trails NSFL,\n"
        "the fused-score signal matters beyond calibration."
    )
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classical t-norm / t-conorm baselines over an NSFL score dump (v2)"
    )
    parser.add_argument("--score_dump", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="debug_out/retrieval")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    run_baselines(args.score_dump, args.out_dir, args.name, args.top_k)
