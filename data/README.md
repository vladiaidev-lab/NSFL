# Data

**This repository does not redistribute any dataset.** All three benchmarks are
obtained from their original sources; nothing in `data/` other than this file is
tracked by git. The MIT license in the repository root covers the code only and
does not apply to any dataset below, each of which keeps its own license.

Scripts expect a `logic_demo/` directory in the working directory (see
`supplementary/README.md` §3 for the full expected layout).

| Dataset | Used for | Source | Redistributed here |
|---|---|---|---|
| QUEST | text-to-text retrieval (Table 4 etc.) | https://github.com/google-research/language/tree/master/language/quest | No — link only |
| COCO-LOGIC | text-to-image retrieval | Released with the paper; built on COCO-2017 val (https://cocodataset.org) | No |
| NevIR | negation (Table 14) | https://huggingface.co/datasets/orionweller/NevIR | No — downloaded automatically by `experiments/evaluate_nevir.py` on first run |

Check each dataset's own license on its source page before reuse.
