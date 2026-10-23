"""
NevIR Evaluation Script for NSFL (v3 - with caching)
=====================================================
Caches dataset and embeddings to disk for fast re-runs.

Usage:
    python evaluate_nevir.py --encoder logic_demo\\models--BAAI--bge-large-en-v1.5 --output debug_out/retrieval/nevir_bge.json
"""

import json
import os
import pickle
import argparse
import numpy as np
from pathlib import Path


NEVIR_BASE_DIR = "logic_demo/NevIR"  # <PATH_TO_NEVIR_DATA>


def get_encoder_short_name(encoder_path: str) -> str:
    """Extract short name from encoder path for folder naming."""
    name = os.path.basename(encoder_path.rstrip("/\\"))
    return name.replace("models--", "").replace("--", "_")


def load_nevir_dataset():
    """Load NevIR dataset, caching to disk on first download."""
    cache_path = os.path.join(NEVIR_BASE_DIR, "dataset")
    
    if os.path.exists(cache_path):
        print(f"Loading cached NevIR dataset from: {cache_path}")
        from datasets import load_from_disk
        ds = load_from_disk(cache_path)
        return ds["test"]
    
    print("Downloading NevIR dataset (first time only)...")
    from datasets import load_dataset
    ds = load_dataset("orionweller/NevIR")
    
    os.makedirs(cache_path, exist_ok=True)
    ds.save_to_disk(cache_path)
    print(f"Dataset cached to: {cache_path}")
    
    return ds["test"]


def encode_and_cache(model, texts, cache_file, label="texts"):
    """Encode texts and cache to disk, or load from cache."""
    if os.path.exists(cache_file):
        print(f"  Loading cached {label} embeddings from: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"  Encoding {label} ({len(texts)} items)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"  Cached to: {cache_file}")
    
    return embeddings


def run_nevir_evaluation(model_path: str, output_path: str):
    # Load model
    print(f"Loading model: {model_path}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, trust_remote_code=True)
    
    # Load dataset
    dataset = load_nevir_dataset()
    print(f"Loaded {len(dataset)} test pairs")
    
    # Setup cache directory
    encoder_name = get_encoder_short_name(model_path)
    cache_dir = os.path.join(NEVIR_BASE_DIR, encoder_name)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Embedding cache dir: {cache_dir}")
    
    # Collect texts
    all_q1 = [item['q1'] for item in dataset]
    all_q2 = [item['q2'] for item in dataset]
    all_doc1 = [item['doc1'] for item in dataset]
    all_doc2 = [item['doc2'] for item in dataset]
    
    # Encode with caching
    print("Encoding (or loading from cache)...")
    emb_q1 = encode_and_cache(model, all_q1, os.path.join(cache_dir, "emb_q1.pkl"), "q1")
    emb_q2 = encode_and_cache(model, all_q2, os.path.join(cache_dir, "emb_q2.pkl"), "q2")
    emb_doc1 = encode_and_cache(model, all_doc1, os.path.join(cache_dir, "emb_doc1.pkl"), "doc1")
    emb_doc2 = encode_and_cache(model, all_doc2, os.path.join(cache_dir, "emb_doc2.pkl"), "doc2")
    
    # Evaluate
    print("Evaluating pairs...")
    
    baseline_correct_q1 = 0
    baseline_correct_q2 = 0
    baseline_correct_both = 0
    
    nsfl_correct_q1 = 0
    nsfl_correct_q2 = 0
    nsfl_correct_both = 0
    
    per_pair = []
    
    for i in range(len(dataset)):
        # Raw cosine similarities
        s_q1_d1 = float(np.dot(emb_q1[i], emb_doc1[i]))
        s_q1_d2 = float(np.dot(emb_q1[i], emb_doc2[i]))
        s_q2_d1 = float(np.dot(emb_q2[i], emb_doc1[i]))
        s_q2_d2 = float(np.dot(emb_q2[i], emb_doc2[i]))
        
        # Baseline correctness
        b_q1_ok = s_q1_d1 > s_q1_d2   # q1 prefers doc1
        b_q2_ok = s_q2_d2 > s_q2_d1   # q2 prefers doc2
        
        if b_q1_ok:
            baseline_correct_q1 += 1
        if b_q2_ok:
            baseline_correct_q2 += 1
        if b_q1_ok and b_q2_ok:
            baseline_correct_both += 1
        
        # NSFL: symmetric delta amplification
        # For q1: amplify difference from q2
        nsfl_q1_d1 = 2 * s_q1_d1 - s_q2_d1
        nsfl_q1_d2 = 2 * s_q1_d2 - s_q2_d2
        
        # For q2: amplify difference from q1
        nsfl_q2_d1 = 2 * s_q2_d1 - s_q1_d1
        nsfl_q2_d2 = 2 * s_q2_d2 - s_q1_d2
        
        # NSFL correctness
        n_q1_ok = nsfl_q1_d1 > nsfl_q1_d2
        n_q2_ok = nsfl_q2_d2 > nsfl_q2_d1
        
        if n_q1_ok:
            nsfl_correct_q1 += 1
        if n_q2_ok:
            nsfl_correct_q2 += 1
        if n_q1_ok and n_q2_ok:
            nsfl_correct_both += 1
        
        per_pair.append({
            "id": dataset[i]["id"],
            "scores": {
                "s_q1_d1": s_q1_d1, "s_q1_d2": s_q1_d2,
                "s_q2_d1": s_q2_d1, "s_q2_d2": s_q2_d2,
            },
            "baseline": {"q1_correct": b_q1_ok, "q2_correct": b_q2_ok},
            "nsfl": {"q1_correct": n_q1_ok, "q2_correct": n_q2_ok},
        })
    
    n = len(dataset)
    
    summary = {
        "model": model_path,
        "encoder_name": encoder_name,
        "n_pairs": n,
        "baseline": {
            "q1_accuracy": baseline_correct_q1 / n,
            "q2_accuracy": baseline_correct_q2 / n,
            "both_accuracy": baseline_correct_both / n,
            "overall_accuracy": (baseline_correct_q1 + baseline_correct_q2) / (2 * n),
        },
        "nsfl": {
            "q1_accuracy": nsfl_correct_q1 / n,
            "q2_accuracy": nsfl_correct_q2 / n,
            "both_accuracy": nsfl_correct_both / n,
            "overall_accuracy": (nsfl_correct_q1 + nsfl_correct_q2) / (2 * n),
        },
    }
    
    b_overall = summary["baseline"]["overall_accuracy"]
    n_overall = summary["nsfl"]["overall_accuracy"]
    summary["improvement"] = n_overall - b_overall
    summary["relative_improvement"] = (n_overall - b_overall) / b_overall if b_overall > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("NevIR Evaluation Results")
    print("=" * 60)
    print(f"Model: {encoder_name}")
    print(f"Test pairs: {n}")
    print()
    print("                    Baseline    NSFL")
    print(f"  q1 accuracy:      {summary['baseline']['q1_accuracy']:.4f}      {summary['nsfl']['q1_accuracy']:.4f}")
    print(f"  q2 accuracy:      {summary['baseline']['q2_accuracy']:.4f}      {summary['nsfl']['q2_accuracy']:.4f}")
    print(f"  both correct:     {summary['baseline']['both_accuracy']:.4f}      {summary['nsfl']['both_accuracy']:.4f}")
    print(f"  overall:          {b_overall:.4f}      {n_overall:.4f}")
    print()
    print(f"  Improvement: {summary['improvement']*100:+.2f}%")
    print("=" * 60)
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    output = {"summary": summary, "per_pair": per_pair}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--output", type=str, default="nevir_results.json")
    args = parser.parse_args()
    
    run_nevir_evaluation(args.encoder, args.output)
    