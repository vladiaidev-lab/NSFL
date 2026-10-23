"""
Experiment B: Direct Empirical Validation of the Three NSFL Design Principles
============================================================================

Reviewer objection: "Several key conjectures are used to motivate the operators
but are not directly evaluated."

This script tests each principle directly against retrieval ground truth, using
the score dump produced by score_dump.ScoreDumper. No re-encoding required.

Principle 1 (NS-Delta Directionality)
    Claim: a positive delta indicates manifold reinforcement, a negative delta
    indicates semantic departure.
    Test: compare P(delta > 0 | relevant) against P(delta > 0 | irrelevant),
    and report the point-biserial correlation between delta and relevance.

Principle 2 (Asymmetric Delta-Sensitivity / Semantic Bottleneck)
    Claim: atomic constituents contribute unequally, and the weaker atom is the
    binding constraint on logical validity.
    Test: (a) show the asymmetry |d_A| - |d_B| is not concentrated at zero;
    (b) show min(d_A, d_B) discriminates relevance better than max(d_A, d_B),
    which is the operative bottleneck claim.

Principle 3 (Negation as Signal Attenuation)
    Claim: negation attenuates delta magnitude while preserving its sign, rather
    than inverting semantics.
    Test: on inhibitory templates, compare the delta induced by the negated
    monolithic query against the delta induced by the affirmative fused query.
    Report the fraction preserving sign and the mean attenuation ratio.

Usage
-----
    python validate_conjectures.py \
        --score_dump debug_out/scores/scores_BGE.tsv \
        --output debug_out/analysis/conjectures_BGE.json
"""

import os
import csv
import json
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats


CONJUNCTIVE_TEMPLATES = {
    "_ that are also _",
    "_ that are also both _ and _",
}

DISJUNCTIVE_TEMPLATES = {
    "_ or _",
    "_ or _ or _",
}

INHIBITORY_TEMPLATES = {
    "_ that are not _",
    "_ that are also _ but not _",
}

TEMPLATE_FAMILY = {
    "conjunctive": CONJUNCTIVE_TEMPLATES,
    "disjunctive": DISJUNCTIVE_TEMPLATES,
    "inhibitory": INHIBITORY_TEMPLATES,
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_rows(path):
    """Load the score dump as a flat list of row dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            def f(key):
                val = row.get(key, "NaN")
                if val is None or val == "" or val == "NaN":
                    return float("nan")
                return float(val)

            rows.append(
                {
                    "qid": row["qid"],
                    "docid": row["docid"],
                    "S_A": f("S_A"),
                    "S_B": f("S_B"),
                    "S_C": f("S_C"),
                    "S_AB": f("S_AB"),
                    "S_ABC": f("S_ABC"),
                    "S_M": f("S_M"),
                    "S_fused_used": f("S_fused_used"),
                    "is_relevant": int(row["is_relevant"]),
                    "template": row["template"],
                }
            )
    return rows


def fused_of(row):
    """
    Return the value the NSFL operator itself actually consumed as its fused
    score for this row.

    Prefers S_fused_used (added to the dump schema specifically because it
    is NOT always equal to S_AB/S_ABC: for pure-AND formulas,
    evaluate_fuzzy_nsfl_clean substitutes the monolithic score in place of
    the computed AB/ABC column, so reading S_ABC there would measure a
    quantity NSFL never used). Falls back to the old S_ABC/S_AB guess only
    for dumps produced before S_fused_used existed.
    """
    if "S_fused_used" in row and not np.isnan(row["S_fused_used"]):
        return row["S_fused_used"]
    if not np.isnan(row["S_ABC"]):
        return row["S_ABC"]
    return row["S_AB"]


# ----------------------------------------------------------------------------
# Principle 1: Delta Directionality
# ----------------------------------------------------------------------------

def _directionality_for_rows(rows):
    """Core Principle 1 computation over a fixed set of rows (one family)."""
    deltas_rel, deltas_irrel = [], []

    for row in rows:
        s_fused = fused_of(row)
        s_a = row["S_A"]
        if np.isnan(s_fused) or np.isnan(s_a):
            continue
        # Delta of the auxiliary atom, conditioned on A
        delta = s_fused - s_a
        if row["is_relevant"]:
            deltas_rel.append(delta)
        else:
            deltas_irrel.append(delta)

    deltas_rel = np.array(deltas_rel)
    deltas_irrel = np.array(deltas_irrel)

    if len(deltas_rel) == 0 or len(deltas_irrel) == 0:
        return {"error": "insufficient data"}

    p_pos_rel = float(np.mean(deltas_rel > 0))
    p_pos_irrel = float(np.mean(deltas_irrel > 0))

    # Point-biserial correlation between delta and the relevance indicator
    all_deltas = np.concatenate([deltas_rel, deltas_irrel])
    all_labels = np.concatenate(
        [np.ones(len(deltas_rel)), np.zeros(len(deltas_irrel))]
    )
    r_pb, p_pb = stats.pointbiserialr(all_labels, all_deltas)

    # Mann-Whitney U: are relevant deltas stochastically larger?
    u_stat, p_mw = stats.mannwhitneyu(
        deltas_rel, deltas_irrel, alternative="greater"
    )

    return {
        "n_relevant": int(len(deltas_rel)),
        "n_irrelevant": int(len(deltas_irrel)),
        "p_positive_delta_given_relevant": p_pos_rel,
        "p_positive_delta_given_irrelevant": p_pos_irrel,
        "difference": p_pos_rel - p_pos_irrel,
        "mean_delta_relevant": float(np.mean(deltas_rel)),
        "mean_delta_irrelevant": float(np.mean(deltas_irrel)),
        "point_biserial_r": float(r_pb),
        "point_biserial_p": float(p_pb),
        "mannwhitney_p": float(p_mw),
    }


def test_directionality(rows):
    """
    Compare the sign of the Neuro-Symbolic Delta between relevant and
    irrelevant candidates, split by connective family.

    IMPORTANT (fixed after an initial mis-specification): pooling all six
    templates confounds the test. For inhibitory templates ("A and not B"),
    the claim's own logic predicts the delta should be NEGATIVE for relevant
    documents - a relevant doc genuinely lacks B, so the affirmative fusion
    S_AB scores it low relative to S_A. That is correct behavior, not a
    refutation, and pooling it in with conjunctive/disjunctive templates
    (where the claim does predict a positive delta for relevant docs) muddies
    a test that is only meaningfully evaluated separately per family. Reports
    all three families; the claim itself is only meaningfully *tested* on
    conjunctive and disjunctive - inhibitory is reported for completeness and
    is expected to show the opposite sign.
    """
    by_family = {}
    for family, templates in TEMPLATE_FAMILY.items():
        family_rows = [r for r in rows if r["template"] in templates]
        by_family[family] = _directionality_for_rows(family_rows)
    return by_family


# ----------------------------------------------------------------------------
# Principle 2: Asymmetric Delta-Sensitivity
# ----------------------------------------------------------------------------

def test_asymmetry(rows):
    """
    Quantify per-atom delta asymmetry, and test whether the minimum delta
    discriminates relevance better than the maximum delta.

    Uses fused_of(row) rather than row["S_AB"] directly, for the same reason
    Principles 1 and 3 do: for "_ that are also _" (pure binary AND),
    row["S_AB"] holds the computed-but-unused AB column, not what NSFL
    actually consumed (S_M). Side effect of this fix: "_ that are also both
    _ and _" (ternary AND) rows, previously excluded entirely because
    row["S_AB"] is always NaN for that template, now contribute too, via
    fused_of() falling through to S_ABC / S_fused_used (= S_M for this
    formula as well, since it is also a pure-AND formula). The delta math
    below remains 2-atom-only (d_a, d_b) even for those ternary rows - atom
    C's contribution is not modeled here; that is a pre-existing gap in this
    test's design, not something this fix addresses.
    """
    asymmetries = []
    min_deltas, max_deltas, labels = [], [], []

    for row in rows:
        if row["template"] not in CONJUNCTIVE_TEMPLATES:
            continue
        s_ab = fused_of(row)
        s_a, s_b = row["S_A"], row["S_B"]
        if np.isnan(s_ab) or np.isnan(s_a) or np.isnan(s_b):
            continue

        # Marginal contribution of each atom given the other
        d_b = s_ab - s_a   # contribution of B, conditioned on A
        d_a = s_ab - s_b   # contribution of A, conditioned on B

        asymmetries.append(abs(abs(d_a) - abs(d_b)))
        min_deltas.append(min(d_a, d_b))
        max_deltas.append(max(d_a, d_b))
        labels.append(row["is_relevant"])

    if len(asymmetries) < 10:
        return {"error": "insufficient conjunctive data"}

    asymmetries = np.array(asymmetries)
    min_deltas = np.array(min_deltas)
    max_deltas = np.array(max_deltas)
    labels = np.array(labels, dtype=float)

    # Discriminative power of each aggregate, measured as |correlation|
    r_min, p_min = stats.pointbiserialr(labels, min_deltas)
    r_max, p_max = stats.pointbiserialr(labels, max_deltas)

    return {
        "n_candidates": int(len(asymmetries)),
        "mean_absolute_asymmetry": float(np.mean(asymmetries)),
        "median_absolute_asymmetry": float(np.median(asymmetries)),
        "frac_asymmetry_above_0.01": float(np.mean(asymmetries > 0.01)),
        "frac_asymmetry_above_0.05": float(np.mean(asymmetries > 0.05)),
        "min_delta_correlation_r": float(r_min),
        "min_delta_correlation_p": float(p_min),
        "max_delta_correlation_r": float(r_max),
        "max_delta_correlation_p": float(p_max),
        "bottleneck_supported": bool(abs(r_min) > abs(r_max)),
    }


# ----------------------------------------------------------------------------
# Principle 3: Negation as Signal Attenuation
# ----------------------------------------------------------------------------

def test_negation_attenuation(rows):
    """
    On inhibitory templates, compare the delta induced by the negated
    monolithic query against the delta induced by the affirmative fused query.

    Attenuation without inversion predicts:
        sign(delta_negated) == sign(delta_fused)  for most candidates
        |delta_negated| < |delta_fused|           on average
    """
    sign_preserved = []
    attenuation_ratios = []
    n_inverted = 0

    for row in rows:
        if row["template"] not in INHIBITORY_TEMPLATES:
            continue
        s_a, s_m = row["S_A"], row["S_M"]
        s_fused = fused_of(row)
        if np.isnan(s_a) or np.isnan(s_m) or np.isnan(s_fused):
            continue

        delta_fused = s_fused - s_a      # affirmative fusion
        delta_negated = s_m - s_a        # negated monolithic query

        if abs(delta_fused) < 1e-9:
            continue

        same_sign = (delta_fused > 0) == (delta_negated > 0)
        sign_preserved.append(same_sign)
        if not same_sign:
            n_inverted += 1

        attenuation_ratios.append(abs(delta_negated) / abs(delta_fused))

    if len(sign_preserved) < 10:
        return {"error": "insufficient inhibitory data (is S_M being dumped?)"}

    sign_preserved = np.array(sign_preserved, dtype=float)
    attenuation_ratios = np.array(attenuation_ratios)

    # A ratio below 1.0 indicates attenuation rather than amplification
    frac_attenuated = float(np.mean(attenuation_ratios < 1.0))

    return {
        "n_candidates": int(len(sign_preserved)),
        "frac_sign_preserved": float(np.mean(sign_preserved)),
        "n_sign_inverted": int(n_inverted),
        "mean_attenuation_ratio": float(np.mean(attenuation_ratios)),
        "median_attenuation_ratio": float(np.median(attenuation_ratios)),
        "frac_attenuated": frac_attenuated,
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def print_report(results):
    """Print a human-readable summary of all three tests."""
    print("\n" + "=" * 68)
    print("NSFL Design Principle Validation")
    print("=" * 68)

    by_family = results["principle_1_directionality"]
    print("\nPrinciple 1: NS-Delta Directionality (split by connective family)")
    print("  Claim is only meaningfully tested on conjunctive/disjunctive;")
    print("  inhibitory is expected to invert by construction - reported, not scored.")
    for family in ("conjunctive", "disjunctive", "inhibitory"):
        d = by_family[family]
        print(f"\n  [{family}]")
        if "error" in d:
            print(f"    {d['error']}")
            continue
        print(f"    P(delta > 0 | relevant)   = {d['p_positive_delta_given_relevant']:.4f}")
        print(f"    P(delta > 0 | irrelevant) = {d['p_positive_delta_given_irrelevant']:.4f}")
        print(f"    difference                = {d['difference']:+.4f}")
        print(f"    mean delta (relevant)     = {d['mean_delta_relevant']:+.5f}")
        print(f"    mean delta (irrelevant)   = {d['mean_delta_irrelevant']:+.5f}")
        print(f"    point-biserial r          = {d['point_biserial_r']:+.4f} "
              f"(p = {d['point_biserial_p']:.2e})")
        print(f"    Mann-Whitney p            = {d['mannwhitney_p']:.2e}")
        if family == "inhibitory":
            print(f"    -> expected inversion (not a test of the claim)")
        else:
            verdict = "SUPPORTED" if d["difference"] > 0 else "NOT SUPPORTED"
            print(f"    -> {verdict}")

    a = results["principle_2_asymmetry"]
    print("\nPrinciple 2: Asymmetric Delta-Sensitivity")
    if "error" in a:
        print(f"  {a['error']}")
    else:
        print(f"  mean |asymmetry|            = {a['mean_absolute_asymmetry']:.5f}")
        print(f"  frac asymmetry > 0.01       = {a['frac_asymmetry_above_0.01']:.4f}")
        print(f"  frac asymmetry > 0.05       = {a['frac_asymmetry_above_0.05']:.4f}")
        print(f"  min-delta correlation       = {a['min_delta_correlation_r']:+.4f}")
        print(f"  max-delta correlation       = {a['max_delta_correlation_r']:+.4f}")
        verdict = "SUPPORTED" if a["bottleneck_supported"] else "NOT SUPPORTED"
        print(f"  -> bottleneck claim {verdict}")

    n = results["principle_3_negation"]
    print("\nPrinciple 3: Negation as Signal Attenuation")
    if "error" in n:
        print(f"  {n['error']}")
    else:
        print(f"  frac sign preserved        = {n['frac_sign_preserved']:.4f}")
        print(f"  n sign inverted            = {n['n_sign_inverted']}")
        print(f"  mean attenuation ratio     = {n['mean_attenuation_ratio']:.4f}")
        print(f"  median attenuation ratio   = {n['median_attenuation_ratio']:.4f}")
        print(f"  frac attenuated (< 1.0)    = {n['frac_attenuated']:.4f}")
        supported = n["frac_sign_preserved"] > 0.5 and n["mean_attenuation_ratio"] < 1.0
        verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
        print(f"  -> {verdict}")

    print("\n" + "=" * 68)
    print(
        "Report these numbers verbatim in the paper. If a principle is NOT\n"
        "supported, say so and soften the corresponding claim — that is a\n"
        "stronger position with TMLR than leaving the claim untested."
    )
    print("=" * 68)


def run(score_dump, output_path):
    """Run all three validation tests and write results to JSON."""
    print(f"Loading score dump: {score_dump}")
    rows = load_rows(score_dump)
    print(f"  {len(rows)} candidate rows loaded")

    results = {
        "score_dump": score_dump,
        "n_rows": len(rows),
        "principle_1_directionality": test_directionality(rows),
        "principle_2_asymmetry": test_asymmetry(rows),
        "principle_3_negation": test_negation_attenuation(rows),
    }

    print_report(results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Directly validate the three NSFL design principles"
    )
    parser.add_argument("--score_dump", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="debug_out/analysis/conjectures.json")
    args = parser.parse_args()

    run(args.score_dump, args.output)
