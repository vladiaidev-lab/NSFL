from datasets import load_dataset


HF_REPO_NAME = "<PATH_TO_DATA>/BoolQuestions"  # only used by get_boolquestions_* below;
                                                # the reported NSFL results use get_fuzzy_evalset()
                                                # with pre-loaded query data instead, not this path


def get_boolquestions_docid2doc(subset_name):
    print("Loading corpus for", subset_name)
    corpus = load_dataset(HF_REPO_NAME, name=f"{subset_name}-corpus", split="corpus")
    if subset_name == "MSMARCO":
        docid2doc = {sample["docid"]: sample["doc"] for sample in corpus}
    elif subset_name == "NaturalQuestions":
        docid2doc = {
            sample["docid"]: f'{sample["title"]} [SEP] {sample["doc"]}'
            for sample in corpus
        }
    else:
        raise ValueError(f"Unsupported BoolQuestions subset name: {subset_name}")
    return docid2doc

def get_fuzzy_evalset(
    eval_data, subset_question_types="all", return_negatives=False, fuzzy=False
):
    print("Loading evaluation set for Fuzzy Logic")
    if isinstance(subset_question_types, str):
        subset_question_types = [subset_question_types]
    if subset_question_types == ["all"]:
        subset_question_types = ["_ or _", "_ that are also both _ and _", "_ or _ or _", "_ that are not _","_ that are also _", "_ that are also _ but not _", "_"]
    qid2query = {}
    qid2docids = {}
    for d in eval_data:
        if d["question_type"] not in subset_question_types:
            continue
        qid = d["qid"]
        if fuzzy:
            qid2query[qid] = d["atoms"] + [["F",d["question"]]]
        else:
            qid2query[qid] = [["F",d["question"]]]
        if return_negatives:
            qid2docids[qid] = [neg_ctx["passage_id"] for neg_ctx in d["negative_ctxs"]]
        else:
            qid2docids[qid] = [pos_ctx["passage_id"] for pos_ctx in d["positive_ctxs"]]
    return qid2query, qid2docids


def get_boolquestions_evalset(
    subset_name, subset_question_types="all", return_negatives=False
):
    print("Loading evaluation set for", subset_name)
    if isinstance(subset_question_types, str):
        subset_question_types = [subset_question_types]
    if subset_question_types == ["all"]:
        subset_question_types = ["and", "or", "not"]
    
    eval_data = load_dataset(HF_REPO_NAME, name=subset_name, split="eval")
    qid2query = {}
    qid2docids = {}
    for d in eval_data:
        if d["question_type"] not in subset_question_types:
            continue
        qid = d["qid"]
        qid2query[qid] = d["question"]
        if return_negatives:
            qid2docids[qid] = [neg_ctx["passage_id"] for neg_ctx in d["negative_ctxs"]]
        else:
            qid2docids[qid] = [pos_ctx["passage_id"] for pos_ctx in d["positive_ctxs"]]
    return qid2query, qid2docids
