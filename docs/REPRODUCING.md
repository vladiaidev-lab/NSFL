# Reproducing the paper's results

Details that the top-level README does not carry. Run every command from the
repository root with `PYTHONPATH=src` (see README, Installation).

<!-- Values marked "TODO: confirm" were not recorded in the package and must be filled in by the author. -->

## Hardware and runtime

- GPU model / memory used for the paper: TODO: confirm
- Per-experiment runtimes: TODO: confirm

## Encoders

Six encoder configurations are used (four text, two vision-language). Model
identifiers that appear in the code or its README:

| Role | Identifier |
|---|---|
| BGE | `BAAI/bge-large-en-v1.5` |
| E5 | `intfloat/e5-base-v2` |
| LogiCoL-E5 | local path `models--...LogiCoL...` (fine-tuned E5; source: TODO: confirm) |
| Nomic | `nomic-ai/nomic-embed-text-v2-moe` |
| BLIP-Large, SigLIP | loaded through `transformers` (`models/models.py`) |

Revisions are **not pinned** in the code: `from_pretrained` / `SentenceTransformer`
resolve to the latest revision unless a local copy is supplied. Exact revisions used
for the paper: TODO: confirm.

## Embeddings and caching

- QUEST/COCO-LOGIC corpora are embedded once into prebuilt FAISS indices
  (`logic_demo/QUEST/faiss_quest_*`, `logic_demo/COCO-LOGIC/faiss_coco_val_*`).
  If an index is absent it is rebuilt on first run (`--rebuild_index`); this is
  the dominant cost. Disk usage per index: TODO: confirm.
- `evaluate_nevir.py` caches query/document embeddings as pickles under
  `logic_demo/NevIR/<encoder>/`, and the dataset under `logic_demo/NevIR/dataset`.

## Key hyperparameters

Defaults in `experiments/retrieve_fuzzy.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--topk` | 100 | documents returned per query (mAP@100) |
| `--topk_max` | 1000 | candidate pool size K that is reranked |
| `--sqo_steps` | 100 | SQO optimisation steps |
| `--sqo_lr` | 0.2 | SQO learning rate |
| `--sqo_patience` | 5 | SQO early-stopping patience |
| `--sqo_n_restarts` | 1 | SQO restarts |
| `--sig_gate` | 0 | significance-gated conjunction (Appendix I, not used for reported results) |

The K ablation (Table 17) sweeps `--topk_max` over {100, 200, 500, 1000}. Values
of per-run overrides used for each reported table: TODO: confirm.

## Splits

QUEST: `val_corpus_no_neg_fixed.jsonl` is the validation split (used for fusion
selection, Tables 15-16); `test_corpus_no_neg_fixed.jsonl` is the test split and
the source of all reported numbers. COCO-LOGIC has no held-out split.
NevIR numbers use the dataset's `test` split (1,383 pairs).

## Known sources of small numerical variation

- Hardware/driver nondeterminism: GPU vs CPU encoding and different batch sizes
  change embeddings in the last digits, which can flip near-tied rankings.
- `transformers` / `sentence-transformers` / `torch` versions, and unpinned model
  revisions, change embeddings.
- FAISS version and index type (flat vs HNSW) affect candidate ordering at ties.
- SQO is initialised randomly; `sqo_seed_variance.py` quantifies the spread
  (Appendix C). The bootstrap CIs in `evaluate.py` use unseeded `numpy` sampling.

Differences in the third or fourth decimal place from the paper are expected from
these sources. Differences larger than that are not, and are worth an issue.

## Known gaps in the code package

Found while preparing this repository; the code was not changed.

- `src/fuzzy_logic.py` imports `pylogics.semantics.ltl.evaluate_ltl` (plus
  `reset_cache`, `print_cache`, `eval_cache`). These do not exist in the PyPI
  `pylogics` (0.2.1). The authors used a locally modified copy, which is not in this
  repository. Until it is added, `retrieve_fuzzy*.py` cannot be imported from a clean
  environment. The NevIR quick start does not depend on it.
- `src/utils/query_builder.py` was missing from the supplementary package and has
  been added from the project tree (unmodified).
- Import check from a fresh clone: with `requirements.txt` installed, `evaluate`, `evaluate_nevir`,
  `baseline_tnorms`, `validate_conjectures`, `fusion_selection` and `sqo_seed_variance` print `--help`;
  `retrieve_fuzzy*.py`, `fuzzy_logic`, `models.models` and `classifiers.nsfl_base` fail only on the
  missing `pylogics` fork above.
- `retrieve_fuzzy.py` also imports `openvino` and `optimum.intel`, which are not
  listed in the supplementary README's `pip install` line.
- `src/classifiers/nsfl_base.py` references `experiments.fuzzy_logic.classifiers.nsfl_utils`
  (`stable_prune_gt_matrix`), which is not shipped.
- Several flags in `supplementary/README.md` do not match the scripts:
  `validate_conjectures.py` takes `--score_dump` (not `--scores_path`),
  `evaluate_nevir.py` takes `--encoder` (not `--queries_path`), and
  `baseline_tnorms.py` requires `--score_dump` and `--name`. The top-level README
  uses the scripts' actual flags.
