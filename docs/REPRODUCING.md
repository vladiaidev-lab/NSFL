# Reproducing the paper's results

Details that the top-level README does not carry. Run every command from the
repository root with `PYTHONPATH=src` (see README, Installation).

## Encoders

Six encoder configurations are used (four text, two vision-language). Model
identifiers that appear in the code or its README:

| Role | Identifier |
|---|---|
| BGE | `BAAI/bge-large-en-v1.5` |
| E5 | `intfloat/e5-base-v2` |
| LogiCoL-E5-v2 | not a public model; see the note below |
| Nomic | `nomic-ai/nomic-embed-text-v2-moe` |
| BLIP-Large, SigLIP | loaded through `transformers` (`models/models.py`) |

LogiCoL published the method, not weights. The LogiCoL-E5-v2 encoder used in the paper is the
authors' own implementation of that method applied to E5, trained locally. No checkpoint is
released with this repository, so this encoder's rows cannot be reproduced from a download. All
other encoders are public Hugging Face models.

Revisions are **not pinned** in the code: `from_pretrained` / `SentenceTransformer`
resolve to the latest revision unless a local copy is supplied.

## Embeddings and caching

- QUEST/COCO-LOGIC corpora are embedded once into prebuilt FAISS indices
  (`logic_demo/QUEST/faiss_quest_*`, `logic_demo/COCO-LOGIC/faiss_coco_val_*`).
  If an index is absent it is rebuilt on first run (`--rebuild_index`); this is
  the dominant cost.
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

The K ablation (Table 17) sweeps `--topk_max` over {100, 200, 500, 1000}.

## Splits

QUEST: `val_corpus_no_neg_fixed.jsonl` is the validation split (used for fusion
selection, Tables 15-16); `test_corpus_no_neg_fixed.jsonl` is the test split and
the source of all reported numbers. COCO-LOGIC has no held-out split. The query files shipped in `data/` are the exact ones used for the published results, so COCO-LOGIC is fully reproducible given a local copy of the COCO-2017 validation images. The image corpus list that `passage_id` indexes into is shipped as `data/coco_corpus_5000_val.json`; point `--images_path` at a folder that contains the images under `val/`.
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

## Differences from `supplementary/`

`supplementary/` is the OpenReview upload, unmodified. `src/` and `experiments/` are the
same code with these removals only (no numerical logic changed):

- The unused `pylogics` linear-temporal-logic path in `src/fuzzy_logic.py` (the
  `pylogics=True` constructor option and its imports). Both drivers always constructed
  `FuzzyLogic(pylogics=False, ...)`, so that path never executed; the `pylogics=False`
  argument was dropped from the two call sites accordingly.
- An unused `query_parser` import in `src/fuzzy_logic.py`.
- The optional OpenVINO / Intel-XPU encoding path (`openvino`, `optimum-intel`): the
  `use_openvino` branches in `src/models/models.py` and the OpenVINO branch of the text
  encoder wrapper in `experiments/retrieve_fuzzy.py`, which is now named `SentenceEncoder`
  and always delegates to `SentenceTransformer`. The removed branches only ran when the
  torch device was `xpu`.
- `src/classifiers/nsfl_base.py` reduced to the encoder interfaces (`IEncoder`,
  `ITextEncoder`, `IImageEncoder`, `IImageTextEncoder`); its dataset/strategy classes were
  never used by any driver.

Path handling also changed for the released COCO-LOGIC data (no scoring, ranking or evaluation
logic touched):

- `experiments/retrieve_fuzzy_blip_siglip_vv.py` now defaults to `data/coco_logic_600_val_{quest_style|simple}.json`
  (chosen by `--augment_type`) and `data/coco_corpus_5000_val.json`, builds its folder and image
  paths with `os.path.join` instead of Windows backslashes, and decides between local files and
  URLs by an explicit `http(s)://` check on `--images_path` (the old `'\\' in images_path` test
  was always true, because a backslash was always appended).
- `experiments/retrieve_fuzzy.py` recognises the renamed COCO-LOGIC query files when it derives a
  cache filename.
- The released query and corpus files hold bare image filenames instead of absolute paths of
  the authors' machine.

Flags in `supplementary/README.md` that do not match the scripts (the top-level README uses
the scripts' actual flags): `validate_conjectures.py` takes `--score_dump` (not
`--scores_path`), `evaluate_nevir.py` takes `--encoder` (not `--queries_path`), and
`baseline_tnorms.py` requires `--score_dump` and `--name`.

## Known issues

- **Shared query-cache filename (`experiments/retrieve_fuzzy.py`).** When it is given a
  COCO-LOGIC query file, both fusion branches of `load_queries_embeddings` build the same
  cache name, `filtered_100_from_all_coco2017_VAL(augment-simple).pkl`. Running Simple and
  Contextual (QUEST-style) fusion against one cache directory (`--saved_queries_path`) can
  therefore load cached query embeddings from the other fusion type. Workaround: use a
  separate cache directory per fusion type, or clear the cache between runs. The code is
  left as released. `experiments/retrieve_fuzzy_blip_siglip_vv.py` names its cache by
  augment type and is not affected.
