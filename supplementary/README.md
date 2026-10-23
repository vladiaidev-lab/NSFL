# NSFL: Neuro-Symbolic Fuzzy Logic Reranking — Supplementary Code

## 1. Overview

This package contains the code used to produce the results reported in the
paper. NSFL (Neuro-Symbolic Fuzzy Logic) is a post-hoc reranking method for
Boolean-logic-over-embeddings retrieval: a query is decomposed into a logical
formula over atomic sub-queries (e.g. `A AND B`, `A AND B AND NOT C`, `A OR
B OR C`), each atom and their fusions are encoded independently, and a
per-template scoring operator recombines the atomic and fused similarity
scores into a single logical relevance score used to rerank a
dense-retrieval candidate pool. The method is evaluated on text-to-text
retrieval (QUEST) and cross-modal image-to-text retrieval (COCO-LOGIC),
across six encoders (four text, two vision-language), and compared against
monolithic-query and geometric-vector baselines, classical t-norm operators,
and a query-side (pre-retrieval) optimization variant (SQO).

## 2. Requirements

```bash
pip install sentence-transformers faiss-cpu datasets numpy scipy pandas torch geoopt transformers Pillow jaxtyping pylogics
```

Python 3.10+ recommended. GPU (CUDA) is optional but strongly recommended for
encoding; `faiss-gpu` may be substituted for `faiss-cpu` if available. The
image-to-text scripts additionally require `torch` with the vision-language
model weights (BLIP-Large, SigLIP) downloadable via `transformers`.

## 3. Data layout

Scripts expect a `logic_demo/` directory at the repository root (create it
next to these scripts, or pass `<PATH_TO_DATA>` explicitly via the relevant
`--*_path` argument). Expected structure:

```
logic_demo/
├── QUEST/
│   ├── documents.jsonl                        # corpus
│   ├── train_corpus_no_neg_fixed.jsonl
│   ├── val_corpus_no_neg_fixed.jsonl           # validation split
│   ├── test_corpus_no_neg_fixed.jsonl          # test split (reported numbers)
│   ├── faiss_quest_bge/                        # prebuilt FAISS index, BGE
│   ├── faiss_quest_zeroshot-e5/                # prebuilt FAISS index, E5-base-v2
│   ├── faiss_quest_logicol-e5/                 # prebuilt FAISS index, LogiCoL-E5-v2
│   └── faiss_quest_nomic-nomic-v2/             # prebuilt FAISS index, Nomic-Embed-v2
├── COCO-LOGIC/
│   ├── coco_corpus_val.json                    # COCO-2017 val image corpus
│   ├── coco_queries_600_val_(augment-quest).json
│   ├── coco_queries_600_val_(augment-simple).json
│   ├── coco_atoms.json
│   └── faiss_coco_val_{blip_large,siglip}/     # prebuilt FAISS indices
├── NevIR/
│   └── dataset/                                # NevIR benchmark data
├── COCO2017/                                   # raw COCO-2017 images, <PATH_TO_DATA>
└── models--<encoder-name>/                     # local HF model caches (optional;
                                                 #  omit to download on first run)
```

FAISS indices are rebuilt automatically on first run if absent (`--rebuild_index`
or if `faiss_data_path` doesn't already contain an index); prebuilding once
and reusing across runs is recommended, since re-embedding the corpus is the
dominant cost.

## 4. Reproducing each table

All commands assume the working directory is this package's root, with
`logic_demo/` present as above. Replace `<ENCODER_PATH>` with a local model
path or a Hugging Face model id (e.g. `BAAI/bge-large-en-v1.5`).

### Table 4 — Main results (NSFL vs. monolithic baseline)

Text encoders (QUEST), one run per encoder/fusion realization:
```bash
python retrieve_fuzzy.py \
  --input_path=logic_demo/QUEST/documents.jsonl \
  --faiss_data_path=logic_demo/QUEST/faiss_quest_bge \
  --queries_path=logic_demo/QUEST/test_corpus_no_neg_fixed.jsonl \
  --query_encoder=<ENCODER_PATH> \
  --saved_queries_path=logic_demo/QUEST/faiss_quest_bge/ \
  --is_decompose=1 --fuzzy_type=NSFL --augment_type=QUEST --name=<run-name>

python evaluate.py \
  --queries_path=logic_demo/QUEST/test_corpus_no_neg_fixed.jsonl \
  --results_file=debug_out/retrieval/results.decompose1_<run-name>.txt
```
The monolithic baseline is the same command with `--is_decompose=0`.

Image encoders (COCO-LOGIC):
```bash
python retrieve_fuzzy_blip_siglip_vv.py \
  --model_name=blip-large --name=<run-name> \
  --is_decompose=1 --augment_type=QUEST

python evaluate.py \
  --queries_path="logic_demo/COCO-LOGIC/coco_queries_600_val_(augment-quest).json" \
  --results_file=debug_out/retrieval/results.decompose1_<run-name>.txt
```
Use `--model_name=siglip` for the SigLIP row.

### Table 5 — Classical t-norm baselines
```bash
python baseline_tnorms.py --help   # see script for per-t-norm invocation
```

### Table 6 + SQO ablation (Opt-only / Hybrid)
Add `--is_opt_onsphere=1` to the `retrieve_fuzzy.py` (or
`retrieve_fuzzy_blip_siglip_vv.py`) command above. `--is_decompose=0` gives
Opt-only (SQO-optimized query vector seeds retrieval, no post-hoc rerank);
`--is_decompose=1` gives Hybrid (SQO-optimized seed, then reranked).
Relevant SQO hyperparameters: `--sqo_steps`, `--sqo_lr`, `--sqo_patience`,
`--sqo_min_step_before_stop`, `--sqo_n_restarts`.

### Tables 12–13 — Statistical significance
```bash
python evaluate.py \
  --queries_path=logic_demo/QUEST/test_corpus_no_neg_fixed.jsonl \
  --results_file=debug_out/retrieval/results.decompose1_<nsfl-run>.txt \
  --baseline_results_file=debug_out/retrieval/results.decompose0_<baseline-run>.txt
```
Writes a Wilcoxon signed-rank test and bootstrap 95% CI per logical template
into the output `.metrics.json`.

### Tables 9–10 — Conjecture validation
Requires a per-candidate score dump first (`--dump_scores_path`, wired into
`retrieve_fuzzy.py`; uses `score_dump.py`), then:
```bash
python retrieve_fuzzy.py ... --dump_scores_path=debug_out/scores/scores_<encoder>.tsv
python validate_conjectures.py --scores_path=debug_out/scores/scores_<encoder>.tsv
```

### Table 14 — NevIR
```bash
python evaluate_nevir.py --queries_path=logic_demo/NevIR/dataset
```

### Tables 15–16 — Fusion selection
```bash
python fusion_selection.py \
  --val_simple  debug_out/retrieval/results.decompose1_<enc>-val-simple.metrics.json \
  --val_quest   debug_out/retrieval/results.decompose1_<enc>-val-quest.metrics.json \
  --test_simple debug_out/retrieval/results.decompose1_<enc>-test-simple.metrics.json \
  --test_quest  debug_out/retrieval/results.decompose1_<enc>-test-quest.metrics.json \
  --name <enc> --output debug_out/analysis/fusion_<enc>.json
```

### Table 17 — K (oversampling pool size) ablation
Re-run `retrieve_fuzzy.py` with `--topk_max` set to each value in
{100, 200, 500, 1000}, then evaluate as in Table 4.

### Appendix C — SQO seed variance
```bash
python sqo_seed_variance.py --help
```

### Appendix I — Significance-gated conjunction (evaluated, not adopted)
```bash
python retrieve_fuzzy.py ... --sig_gate=1
```
See Notes below — this flag is off by default and is not used for any
reported result.

## 5. Notes — things a reader will otherwise find surprising

1. **Conjunctive operators anchor on the monolithic score, not a separately
   encoded fusion.** For the two pure-conjunctive templates (`A AND B`,
   `A AND B AND C`), the reranking operator substitutes the monolithic query
   embedding's similarity score ($S_M$) in place of the separately-encoded
   fused pseudo-atom score ($S_{AB}$/$S_{ABC}$) that is otherwise computed
   for every other template. This is intentional (see paper §6.2 and
   Appendix A) and is not a bug — the fused score for these two templates is
   still computed and can be inspected via the score dumper, but the
   operator itself does not consume it.
2. **`--sig_gate` is included for completeness but is not used for any
   reported result.** It implements a significance-gated variant of the
   conjunctive correction (scaling the correction term by its magnitude
   relative to the per-query candidate-pool noise floor), evaluated during
   revision as a possible improvement. It did not produce a reliable
   accuracy gain on the held-out test split and was not adopted; it is
   included, off by default, purely so the ablation in Appendix I is
   reproducible.

## 6. Correspondence to paper sections

| Script | Paper section / appendix |
|---|---|
| `fuzzy_logic.py` | §6 (Logic Operators and Scoring Methods), Appendix A (S_M substitution), Appendix I (`--sig_gate`) |
| `objective_hnsw.py` | §7 (SQO: Sphere Query Optimization) |
| `retrieve_fuzzy.py` | §5 (Pipeline), Table 4 (text), Table 6 + SQO ablation, Table 17 (K ablation) |
| `retrieve_fuzzy_blip_siglip_vv.py` | Table 4 (image-to-text), cross-modal results |
| `models/models.py`, `models/blip/blip_model.py` | Vision-language encoder wrappers (BLIP-Large, SigLIP) used by the above |
| `dataset.py` | Query/corpus loading (`get_fuzzy_evalset`) used by `evaluate.py` |
| `evaluate.py` | Tables 4, 12–13 (statistical significance) |
| `score_dump.py` | Per-candidate score export, feeds Tables 9–10 |
| `baseline_tnorms.py` | Table 5 (classical t-norm baselines) |
| `validate_conjectures.py` | Tables 9–10 (design-principle conjecture validation) |
| `fusion_selection.py` | Tables 15–16 (fusion realization selection protocol) |
| `evaluate_nevir.py` | Table 14 (NevIR) |
| `sqo_seed_variance.py` | Appendix C (SQO seed/initialization variance) |
