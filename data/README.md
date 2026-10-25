# Data

This repository redistributes one dataset, COCO-LOGIC: the query set, its ground truth and an
image corpus list. The images themselves come from COCO. QUEST and NevIR are obtained from
their original sources.

<!-- TODO: licensor / copyright holder "Huawei Technologies Co., Ltd." pending confirmation -->
The MIT license in the repository root covers the code only. The COCO-LOGIC query set and
ground truth are released under CC BY 4.0 (licence text and attribution in [`LICENSE`](LICENSE)
in this directory, which covers those files only). The image corpus list (`coco_corpus_5000_val.json`)
is a re-formatted listing of COCO-2017 validation filenames, provided for convenience; it
contains no original content and is covered by COCO's own terms
(https://cocodataset.org/#termsofuse). COCO images and annotations are not redistributed and
keep their own licenses from the original source. QUEST and NevIR keep their own licenses.

Scripts expect a `logic_demo/` directory in the working directory. It is still needed for the
FAISS indices, the COCO images, QUEST and NevIR (see `supplementary/README.md` §3 for that
layout). The COCO-LOGIC query and corpus files now live in this directory (`data/`), not under
`logic_demo/COCO-LOGIC/` as that layout describes, so put them here.

| Dataset | Used for | Source | Redistributed here |
|---|---|---|---|
| QUEST | text-to-text retrieval (Table 4 etc.) | https://github.com/google-research/language/tree/master/language/quest | No — link only |
| COCO-LOGIC | text-to-image retrieval | Built on COCO-2017 val (https://cocodataset.org) | Yes — query set, ground truth and image list; images from COCO |
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
`original_query` is provided for readers: the shipped scripts do not read it, and the fusion
wording they encode is set by `--augment_type`, so both files give identical inputs to the
shipped code. `question_type` is the template label used to group queries and report results
per template; the operator is selected from `atom_formulas`. It is identical across the two
files by design: in the simple-fusion file `original_query` reads "A and B" while
`question_type` still reads `"_ that are also _"`. This is correct, not a bug, and
`question_type` must not be edited, since evaluation filters and reports by it.

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
field is empty. Its content is unchanged: it is not filtered or subsetted. It is provided for
convenience, contains no original content, and is covered by COCO's terms of use
(https://cocodataset.org/#termsofuse), not by CC BY 4.0.

**Ground truth.** `passage_id` in the query files is a **positional index** into the corpus
list: `corpus[int(passage_id)]["title"]` equals the `answer` of that `positive_ctxs` entry (this
holds for all entries: 101,001 in each query file, 202,002 in total). The order of the corpus list therefore must
not be changed: do not sort, deduplicate, filter or reorder it. `answer`, `title` and `image`
are COCO-2017 val filenames with no directory, to be resolved against a local copy of the
images; `experiments/retrieve_fuzzy_blip_siglip_vv.py` reads image `<title>` from
`<images_path>/val/` (see its `--images_path`). COCO-LOGIC has no held-out split.
