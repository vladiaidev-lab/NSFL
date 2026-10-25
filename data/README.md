# Data

This repository redistributes one dataset: the COCO-LOGIC query set (queries and ground
truth only; the images come from COCO). QUEST and NevIR are obtained from their original
sources.

The MIT license in the repository root covers the code only. The COCO-LOGIC query set is
released under CC BY 4.0. COCO images and annotations are not redistributed and keep their
own licenses from the original source (https://cocodataset.org). QUEST and NevIR keep their
own licenses.

Scripts expect a `logic_demo/` directory in the working directory (see
`supplementary/README.md` §3 for the full expected layout).

| Dataset | Used for | Source | Redistributed here |
|---|---|---|---|
| QUEST | text-to-text retrieval (Table 4 etc.) | https://github.com/google-research/language/tree/master/language/quest | No — link only |
| COCO-LOGIC | text-to-image retrieval | Built on COCO-2017 val (https://cocodataset.org) | Yes — queries and ground truth only; images from COCO |
| NevIR | negation (Table 14) | https://huggingface.co/datasets/orionweller/NevIR | No — downloaded automatically by `experiments/evaluate_nevir.py` on first run |

Check each dataset's own license on its source page before reuse.

## COCO-LOGIC query files

Two files, each holding the same 600 queries (100 for each of 6 logical templates) over the
COCO-2017 validation images:

| File | Surface realization of the conjunctive templates |
|---|---|
| `coco_logic_600_val_quest_style.json` | QUEST-style contextual fusion: "A that are also B" |
| `coco_logic_600_val_simple.json` | Simple fusion: "A and B" |

Each file is a JSON list of records with the fields `qid`, `question`, `original_query`,
`atoms`, `atom_formulas`, `question_type`, `positive_ctxs` (a list of `{passage_id, answer}`)
and `negative_ctxs` (present, always empty).

**Field roles.** The two files share `qid`, `question`, `atoms`, `atom_formulas`, the ground
truth and `question_type`. They differ **only** in `original_query`, the surface realization
of the same logical template; the two disjunctive templates ("A or B", "A or B or C") are
worded identically in both files, so `original_query` differs for 400 of the 600 queries.
`question_type` is the template identifier that the code dispatches on to select the
operator. It is identical across the two files by design: in the simple-fusion file
`original_query` reads "A and B" while `question_type` still reads `"_ that are also _"`.
This is correct, not a bug, and `question_type` must not be edited.

| `question_type` | `atom_formulas` |
|---|---|
| `_ that are also _` | `atom0 & atom1` |
| `_ that are also both _ and _` | `atom0 & atom1 & atom2` |
| `_ that are also _ but not _` | `atom0 & atom1 & !atom2` |
| `_ that are not _` | `atom0 & !atom1` |
| `_ or _` | `atom0 \| atom1` |
| `_ or _ or _` | `atom0 \| atom1 \| atom2` |

**Ground truth.** `passage_id` identifies an image by its position in the 5,000-image COCO-2017
val corpus list that the scripts read (`logic_demo/COCO-LOGIC/coco_corpus_val.json`, which is
not part of this release). `answer` is the COCO-2017 val image filename, for example
`000000419974.jpg`, to be resolved against a local copy of the images; it carries no directory.
COCO-LOGIC has no held-out split.
