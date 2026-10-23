### EVALUATE.PY

import json
import argparse
from dataset import get_boolquestions_evalset, get_fuzzy_evalset
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import numpy as np
import pandas as pd
import os
from scipy.stats import wilcoxon

def dump_to_csv(all_metrics, test_name, csv_name):
    #df = pd.DataFrame.from_dict(all_metrics)
    results_stats = {'name': test_name}
    results_summary = {'name': test_name}
    for formula in all_metrics:
        for metric in all_metrics[formula]:            
            #metric_mean = np.mean(list(all_metrics[formula][metric].values()))
            key = formula+'___'+metric
            #results[key] = [metric_mean]
            results_stats[key] = [list(all_metrics[formula][metric].values())]
            #if metric in 'ndcg@k,mrr@k,map@k':
            if metric in 'map@k':
                results_summary[key] = list(all_metrics[formula][metric].values())
    
    # save summary report
    df_summary = pd.DataFrame(results_summary)
    csv_summary = csv_name + "_summary.csv"
    if os.path.exists(csv_summary):
        df_summary.to_csv(csv_summary, index=False, mode='a', header=False)#, sheet_name='summary')        
    else:
        df_summary.to_csv(csv_summary, index=False, mode='a')#, sheet_name='summary')

    # save stats report
    df_stats = pd.DataFrame(results_stats)
    csv_stats = csv_name + "_stats.csv"
    if os.path.exists(csv_stats):
        df_stats.to_csv(csv_stats, index=False, mode='a', header=False)       
    else:
        df_stats.to_csv(csv_stats, index=False, mode='a')



def compute_per_query_ap(queries_result_list, evaluator):
    """Compute Average Precision per query."""
    per_query_ap = {}
    
    for query_idx, query_id in enumerate(evaluator.queries_ids):
        results = queries_result_list[query_idx]
        relevant_docs = set(evaluator.relevant_docs.get(query_id, []))
        
        if not relevant_docs or not results:
            per_query_ap[query_id] = 0.0
            continue
        
        # Sort by score descending (already sorted from results file by rank)
        num_relevant = 0
        sum_precision = 0.0
        
        for rank, result in enumerate(results, start=1):
            doc_id = result["corpus_id"]
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                sum_precision += precision_at_rank
        
        if len(relevant_docs) > 0:
            per_query_ap[query_id] = sum_precision / len(relevant_docs)
        else:
            per_query_ap[query_id] = 0.0
    
    return per_query_ap


def compute_statistical_significance(baseline_aps, nsfl_aps, query_ids):
    """Compute Wilcoxon signed-rank test and bootstrap CI."""
    # Align scores by query_id
    common_qids = set(baseline_aps.keys()) & set(nsfl_aps.keys())
    
    baseline_scores = np.array([baseline_aps[qid] for qid in common_qids])
    nsfl_scores = np.array([nsfl_aps[qid] for qid in common_qids])
    
    # Compute differences
    differences = nsfl_scores - baseline_scores
    mean_improvement = np.mean(differences)
    
    # Wilcoxon signed-rank test (one-sided: NSFL > baseline)
    try:
        # Remove zero differences for Wilcoxon
        non_zero_mask = differences != 0
        if np.sum(non_zero_mask) >= 10:  # Need sufficient non-zero pairs
            stat, p_value = wilcoxon(differences[non_zero_mask], alternative='greater')
        else:
            stat, p_value = np.nan, np.nan
    except Exception as e:
        stat, p_value = np.nan, np.nan
    
    # Bootstrap 95% CI
    n_bootstrap = 10000
    bootstrap_means = []
    n = len(differences)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        bootstrap_means.append(np.mean(differences[idx]))
    
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        "n_queries": len(common_qids),
        "mean_improvement": float(mean_improvement),
        "wilcoxon_statistic": float(stat) if not np.isnan(stat) else None,
        "wilcoxon_p_value": float(p_value) if not np.isnan(p_value) else None,
        "significant_p05": bool(p_value < 0.05) if not np.isnan(p_value) else None,
        "significant_p01": bool(p_value < 0.01) if not np.isnan(p_value) else None,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "baseline_mean_ap": float(np.mean(baseline_scores)),
        "nsfl_mean_ap": float(np.mean(nsfl_scores))
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    '--results_file', type=str, default='debug_out/retrieval/results.decompose0_.txt', help="Path to the results file.")
parser.add_argument(
    '--data', type=str, default='MSMARCO',
    help="Name or path of the raw data.")
parser.add_argument(
        "--eval_fuzzy_queries", type=int, default='1',
        help="Whether to encode queries online.")
parser.add_argument(
        "--queries_path", type=str, default='logic_demo/QUEST/val_corpus_no_neg_fixed.jsonl',
        help="Path to the queries file.")
parser.add_argument(
    '--baseline_results_file', type=str, default=None,
    help="Path to baseline results file for statistical significance testing.")

args = parser.parse_args()

all_metrics = {}
all_significance = {}

for question_type in [ "_ or _", "_ that are also both _ and _", "_ or _ or _", "_ that are not _","_ that are also _", "_ that are also _ but not _"]:#, "_","_ or _ but not _"]:
    #for split in ["pos", "neg"]:
    for split in ["pos"]:
        if args.eval_fuzzy_queries:
            print("Loading queries from:", args.queries_path)
            with open(args.queries_path, 'r', encoding='utf-8') as json_data:
                queries = json.load(json_data)
                json_data.close()
            eval_qid2query, eval_qid2docids = get_fuzzy_evalset(queries, subset_question_types=question_type, return_negatives=True if split == "neg" else False)
        else:
            eval_qid2query, eval_qid2docids = get_boolquestions_evalset(
                subset_name=args.data, subset_question_types=question_type, 
                return_negatives=True if split == "neg" else False
            )

        print("Evaluating question types:", question_type)
        print("Number of evaluated questions:", len(eval_qid2docids))

        evaluator = InformationRetrievalEvaluator(queries=eval_qid2query,
                                                corpus={},
                                                relevant_docs=eval_qid2docids,
                                                accuracy_at_k=[1, 3, 5, 10, 20, 100],
                                                precision_recall_at_k=[20, 50, 100, 1000],
                                                #accuracy_at_k=[50],
                                                mrr_at_k=[10, 100],
                                                ndcg_at_k=[10, 100])
        qid2idx = {qid: idx for idx, qid in enumerate(evaluator.queries_ids)}
        queries_result_list = [[] for _ in range(len(qid2idx))]
        with open(args.results_file) as f:
            for line in f:
                qid, docid, rank, score = line.strip().split("\t")
                if qid in qid2idx:
                    queries_result_list[qid2idx[qid]].append({"corpus_id": docid, "score": float(1)})

        metrics = evaluator.compute_metrics(queries_result_list)
        all_metrics[f"{question_type}_{split}"] = metrics

        # Statistical significance testing (if baseline provided)
        if args.baseline_results_file:
            # Load baseline results
            baseline_result_list = [[] for _ in range(len(qid2idx))]
            with open(args.baseline_results_file) as f:
                for line in f:
                    qid, docid, rank, score = line.strip().split("\t")
                    if qid in qid2idx:
                        baseline_result_list[qid2idx[qid]].append({"corpus_id": docid, "score": float(score)})
            
            # Compute per-query AP for both
            nsfl_aps = compute_per_query_ap(queries_result_list, evaluator)
            baseline_aps = compute_per_query_ap(baseline_result_list, evaluator)
            
            # Compute significance
            sig_results = compute_statistical_significance(baseline_aps, nsfl_aps, evaluator.queries_ids)
            all_significance[f"{question_type}_{split}"] = sig_results
            
            if sig_results['wilcoxon_p_value'] is not None:
                print(f"  Significance: p={sig_results['wilcoxon_p_value']:.4f}, "
                    f"CI=[{sig_results['ci_95_lower']:.4f}, {sig_results['ci_95_upper']:.4f}]")
            else:
                print(f"  Significance: insufficient data for Wilcoxon test, "
                    f"CI=[{sig_results['ci_95_lower']:.4f}, {sig_results['ci_95_upper']:.4f}]")

# Add significance to metrics if computed
if args.baseline_results_file:
    all_metrics["_statistical_significance"] = all_significance

with open(args.results_file.replace(".txt", ".metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

results_csv = 'queries_results'
test_name = os.path.splitext(os.path.basename(args.results_file))[0]
dump_to_csv(all_metrics, test_name, results_csv)

from pprint import pprint
pprint(all_metrics)
