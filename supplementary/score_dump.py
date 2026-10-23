"""
Score Dump Helper for NSFL Analysis Experiments  (v2 — corrected)
================================================================

Change from v1: add_query now accepts BOTH fused scores separately
(fused_ab_scores and fused_abc_scores) instead of a single fused_scores
array. This matters for the mixed template "A and B and not C", whose NSFL
operator uses S_AB and S_ABC simultaneously — v1 could only record one of
them, which would have silently broken the K-ablation for that template.

Writes per-candidate atomic and fused similarity scores to a TSV. All
downstream analyses (t-norm baselines, conjecture validation, K ablation)
read from this dump, so retrieval runs once per encoder.

INTEGRATION INTO retrieve_fuzzy.py
----------------------------------

1. CLI argument:

    parser.add_argument('--dump_scores_path', type=str, default=None,
                        help="If set, dump per-candidate scores to this TSV.")

2. Before the query loop:

    from score_dump import ScoreDumper
    dumper = ScoreDumper(args.dump_scores_path) if args.dump_scores_path else None

3. Inside the per-query loop, once atomic / fused / monolithic scores for the
   topk_max candidates exist:

    if dumper is not None:
        dumper.add_query(
            qid=qid,
            doc_ids=monolitic_ids_list,          # candidate ids, rank order
            atomic_scores=atomic_scores_matrix,  # (n_atoms, n_candidates)
            fused_ab_scores=s_ab_array,          # S_AB, or None if not computed
            fused_abc_scores=s_abc_array,        # S_ABC, or None if not computed
            monolithic_scores=monolithic_scores, # S_M, or None
            relevant_doc_ids=ground_truth_ids,
            template=question_type,
        )

   Which fused arrays exist depends on the template:
     - binary templates (A∧B, A∧¬B, A∨B): fused_ab only, fused_abc=None
     - pure ternary (A∧B∧C, A∨B∨C): fused_abc only, fused_ab=None
     - mixed (A∧B∧¬C): BOTH — the operator consumes S_AB and S_ABC together.
   Pass None for whatever the pipeline does not compute; do NOT recompute
   anything just to fill a column.

4. After the loop:

    if dumper is not None:
        dumper.close()

This adds no computation to the retrieval path — only a file write.
"""

import os
import numpy as np


class ScoreDumper:
    """Accumulates and writes per-candidate score rows to a TSV file."""

    HEADER = [
        "qid", "docid", "rank",
        "S_A", "S_B", "S_C",
        "S_AB", "S_ABC", "S_M", "S_fused_used",
        "is_relevant", "template",
    ]

    def __init__(self, path: str):
        """
        Args:
            path: Output TSV file path. Parent directories are created.
        """
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._fh.write("\t".join(self.HEADER) + "\n")
        self.n_rows = 0
        self.n_queries = 0

    @staticmethod
    def _fmt(value):
        """Format a scalar for TSV output, using NaN for missing values."""
        if value is None:
            return "NaN"
        v = float(value)
        if np.isnan(v):
            return "NaN"
        return f"{v:.6f}"

    @staticmethod
    def _as_array(values):
        """Coerce to a float array, or return None if values is None."""
        if values is None:
            return None
        return np.asarray(values, dtype=float).ravel()

    def add_query(
        self,
        qid,
        doc_ids,
        atomic_scores,
        fused_ab_scores,
        fused_abc_scores,
        monolithic_scores,
        relevant_doc_ids,
        template,
        fused_used_scores=None,
    ):
        """
        Record all candidates for a single query.

        Args:
            qid: Query id (str or int).
            doc_ids: Candidate document ids, in rank order.
            atomic_scores: Array (n_atoms, n_candidates). Row 0 is atom A,
                row 1 atom B, row 2 (if present) atom C.
            fused_ab_scores: Array (n_candidates,) of S_AB, or None if the
                pipeline does not compute S_AB for this template.
            fused_abc_scores: Array (n_candidates,) of S_ABC, or None.
            monolithic_scores: Array (n_candidates,) of S_M, or None.
            relevant_doc_ids: Ground-truth relevant document ids.
            template: Template string identifying the logical form.
            fused_used_scores: Array (n_candidates,) of S_fused_used, or None.
                The exact scalar value the NSFL operator itself consumed as
                its fused score for this template - NOT necessarily S_AB or
                S_ABC. For pure-AND formulas (atom0 & atom1,
                atom0 & atom1 & atom2), evaluate_fuzzy_nsfl_clean substitutes
                the monolithic score in place of the computed AB/ABC column,
                so S_fused_used = S_M there even though S_AB/S_ABC are also
                populated with the (unused) computed column. Everywhere else
                it equals the computed column NSFL actually reads. Callers
                that omit this (older integration) get NaN here - downstream
                analyses that depend on it should treat NaN as "not captured
                by this dump", not as "NSFL used nothing".
        """
        atomic_scores = np.atleast_2d(np.asarray(atomic_scores, dtype=float))
        n_atoms, n_cand = atomic_scores.shape

        s_ab_arr = self._as_array(fused_ab_scores)
        s_abc_arr = self._as_array(fused_abc_scores)
        s_m_arr = self._as_array(monolithic_scores)
        s_used_arr = self._as_array(fused_used_scores)

        relevant = {str(d) for d in (relevant_doc_ids or [])}

        for j in range(min(n_cand, len(doc_ids))):
            docid = str(doc_ids[j])

            s_a = atomic_scores[0, j] if n_atoms > 0 else None
            s_b = atomic_scores[1, j] if n_atoms > 1 else None
            s_c = atomic_scores[2, j] if n_atoms > 2 else None

            s_ab = s_ab_arr[j] if s_ab_arr is not None else None
            s_abc = s_abc_arr[j] if s_abc_arr is not None else None
            s_m = s_m_arr[j] if s_m_arr is not None else None
            s_used = s_used_arr[j] if s_used_arr is not None else None

            row = [
                str(qid),
                docid,
                str(j + 1),
                self._fmt(s_a),
                self._fmt(s_b),
                self._fmt(s_c),
                self._fmt(s_ab),
                self._fmt(s_abc),
                self._fmt(s_m),
                self._fmt(s_used),
                "1" if docid in relevant else "0",
                str(template),
            ]
            self._fh.write("\t".join(row) + "\n")
            self.n_rows += 1

        self.n_queries += 1

    def close(self):
        """Flush and close the dump file."""
        self._fh.close()
        print(
            f"[ScoreDumper] Wrote {self.n_rows} rows "
            f"({self.n_queries} queries) to {self.path}"
        )
