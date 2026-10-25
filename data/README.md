# Data

The only dataset redistributed by this repository is the COCO-LOGIC query set
(queries and ground truth; the images come from COCO). QUEST and NevIR are obtained
from their original sources.

The MIT license in the repository root covers the code only. The COCO-LOGIC query set
is released under CC BY 4.0. QUEST and NevIR keep their own licenses.

Scripts expect a `logic_demo/` directory in the working directory (see
`supplementary/README.md` §3 for the full expected layout).

| Dataset | Used for | Source | Redistributed here |
|---|---|---|---|
| QUEST | text-to-text retrieval (Table 4 etc.) | https://github.com/google-research/language/tree/master/language/quest | No — link only |
| COCO-LOGIC | text-to-image retrieval | Built on COCO-2017 val (https://cocodataset.org) | Yes (queries and ground truth only; images from COCO) |
| NevIR | negation (Table 14) | https://huggingface.co/datasets/orionweller/NevIR | No — downloaded automatically by `experiments/evaluate_nevir.py` on first run |

Check each dataset's own license on its source page before reuse.
