# Data

This repository redistributes one dataset: COCO-LOGIC (the query set, its ground truth and
the image list; the images themselves come from COCO). QUEST and NevIR are obtained from their original
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

## COCO-LOGIC files

Three files in `data/`:

| File | Content |
|---|---|
| `coco_logic_600_val_quest_style.json` | 600 queries, QUEST-style contextual fusion: "A that are also B" |
| `coco_logic_600_val_simple.json` | the same 600 queries, simple fusion: "A and B" |
| `coco_corpus_5000_val.json` | the 5,000-image corpus list that `passage_id` indexes into |

The two query files each hold 600 queries (100 for each of 6 logical templates) over the
COCO-2017 validation images.

Each query file is a JSON list of records with the fields `qid`, `question`, `original_query`,
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

**Image corpus list.** `coco_corpus_5000_val.json` is the COCO-2017 validation file list in a
different JSON structure: a list of 5,000 `{title, test, image}` entries whose `title` and
`image` hold the same COCO-2017 val filename (for example `000000212226.jpg`) and whose `test`
field is empty. Its content is unchanged: it is not filtered or subsetted.

**Ground truth.** `passage_id` in the query files is a **positional index** into the corpus
list: `corpus[int(passage_id)]["title"]` equals the `answer` of that `positive_ctxs` entry (this
holds for all 101,001 entries of both query files). The order of the corpus list therefore must
not be changed: do not sort, deduplicate, filter or reorder it. `answer`, `title` and `image`
are COCO-2017 val filenames with no directory, to be resolved against a local copy of the
images; `experiments/retrieve_fuzzy_blip_siglip_vv.py` reads image `<title>` from
`<images_path>/val/` (see its `--images_path`). COCO-LOGIC has no held-out split.
