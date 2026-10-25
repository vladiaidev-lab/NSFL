# NSFL: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<!-- TODO: replace with https://jmlr.org/tmlr/papers/... once published -->
[TMLR paper](TMLR_URL_PLACEHOLDER) · [OpenReview](https://openreview.net/forum?id=WYpsWBabCu)

**Authors:** Vladi Vexler\*, Ofer Idan, Gil Lederman, Dima Sivov
**Affiliation:** Huawei Tel-Aviv Research Center
\*Corresponding author: vladi dot vexler at huawei dot com

## Abstract

Standard dense retrievers have no native mechanism for multi-atom logical constraints. We introduce **Neuro-Symbolic Fuzzy Logic (NSFL)**, a training-free framework that brings fuzzy-logical structure to neural embedding spaces through a first-order finite-difference treatment of similarity. NSFL anchors logical operations on zero-order atomic scores while steering with **Neuro-Symbolic Deltas (NS-Δ)** — the marginal semantic contributions induced by contextual fusion — preserving atomic meaning while capturing contextual dependence, and avoiding the representation collapse and manifold escape endemic to geometric baselines. For latency-constrained deployment, **Spherical Query Optimization (SQO)** uses Riemannian optimization to project logical formulas into manifold-stable query vectors. Across six encoder configurations and two modalities, NSFL yields mAP@100 improvements up to **+81%** on QUEST and **+70%** on a COCO-derived text-to-image benchmark, and provides an additive **+19% average** gain on an encoder already fine-tuned for logical retrieval. On NevIR, where absolute accuracies are directly interpretable, NSFL raises pairwise negation accuracy from **0.611 to 0.717**. The decomposition is exact and general; the operators built on it are engineering realizations rather than a closed algebra, and we state that boundary precisely.

## What this repository contains

Code for the NSFL operators and scoring, Spherical Query Optimization (SQO), and the evaluation drivers for QUEST, COCO-LOGIC and NevIR. `supplementary/` is the code package exactly as uploaded to OpenReview for review; `src/` and `experiments/` contain the same files regrouped (library modules vs. drivers), with only unused code removed (see `docs/REPRODUCING.md`). No dataset is redistributed.

```
.
├── README.md, LICENSE, CITATION.cff, requirements.txt
├── src/                 # operators and scoring (fuzzy_logic.py), SQO (objective_hnsw.py),
│   │                    # score dumping, dataset loading, encoder wrappers
│   ├── models/  classifiers/
├── experiments/         # retrieval / evaluation drivers (retrieve_fuzzy*.py, evaluate*.py, ...)
├── supplementary/       # the OpenReview supplementary package, unmodified
├── data/README.md       # how to obtain the datasets
└── docs/REPRODUCING.md  # hardware, hyperparameters, splits, numerical variation
```

## Installation

Tested with Python 3.12.3 on Linux (WSL2), CPU only.

```bash
git clone https://github.com/vladiaidev-lab/NSFL.git && cd NSFL
python3 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

## Quick start

NevIR with `BAAI/bge-large-en-v1.5` (Table 14). The dataset and model are downloaded automatically; no manual data setup is needed. Embeddings are cached under `logic_demo/NevIR/`.

```bash
PYTHONPATH=src python experiments/evaluate_nevir.py --encoder BAAI/bge-large-en-v1.5 --output debug_out/retrieval/nevir_bge.json
```

Expected: `overall` accuracy of 0.611 (baseline) and 0.717 (NSFL), as reported in the abstract and Table 14.

## Reproducing the paper's results

The commands below are taken from the supplementary package (flags checked against each script's argument parser) and need the QUEST / COCO-LOGIC data (see Datasets); details in [`docs/REPRODUCING.md`](docs/REPRODUCING.md). Run from the repository root with `PYTHONPATH=src`; the scripts expect a `logic_demo/` directory (layout in `supplementary/README.md` §3).

| Paper item | Command |
|---|---|
| Table 4 (QUEST) | `python experiments/retrieve_fuzzy.py ... --is_decompose=1 --fuzzy_type=NSFL --augment_type=QUEST`, then `python experiments/evaluate.py --queries_path=... --results_file=...` |
| Table 4 (COCO-LOGIC) | `python experiments/retrieve_fuzzy_blip_siglip_vv.py --model_name=blip-large\|siglip --is_decompose=1 --augment_type=QUEST`, then `experiments/evaluate.py` |
| Table 5 (t-norm baselines) | `python experiments/baseline_tnorms.py --score_dump=... --name=...` |
| Table 6, SQO ablation | add `--is_opt_onsphere=1` to the Table 4 commands |
| Tables 9–10 | `retrieve_fuzzy.py ... --dump_scores_path=...`, then `python experiments/validate_conjectures.py --score_dump=...` |
| Tables 12–13 | `experiments/evaluate.py ... --baseline_results_file=...` |
| Table 14 (NevIR) | Quick start above |
| Tables 15–16 | `python experiments/fusion_selection.py ...` |
| Table 17 (K ablation) | `retrieve_fuzzy.py ... --topk_max={100,200,500,1000}` |
| Appendix C (SQO seeds) | `python experiments/sqo_seed_variance.py --help` |

## Datasets

- **QUEST** — external, not redistributed; see [`data/README.md`](data/README.md).
- **COCO-LOGIC** — released with the paper, derived from COCO-2017 val; not redistributed in this repository.
- **NevIR** — external, downloaded automatically from Hugging Face on first run.

The MIT license covers the code only; each dataset keeps its own license (see `data/README.md`).

## Citation

```bibtex
@article{vexler2026nsfl,
  title   = {{NSFL}: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings},
  author  = {Vexler, Vladi and Idan, Ofer and Lederman, Gil and Sivov, Dima},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=WYpsWBabCu}
}
```

## License

<!-- TODO: LICENSE pending legal review; may be replaced -->
See [`LICENSE`](LICENSE) (MIT).

## Acknowledgments

This work was carried out at Huawei Tel-Aviv Research Center.
