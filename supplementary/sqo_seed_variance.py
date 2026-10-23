"""
SQO Seed Variance Analysis Script
=================================

Measures stability of Spherical Query Optimization across random seeds.
Runs SQO with multiple random initializations and reports variance.

Usage:
    python sqo_seed_variance.py --encoder bge-large-v1.5 --n_seeds 5 --n_queries 100
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
import time


def riemannian_gradient_descent(
    f_score,
    f_grad,
    d: int,
    lr: float = 0.2,
    max_steps: int = 100,
    patience: int = 10,
    tol: float = 1e-6,
    seed: int = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Riemannian SGD on the unit sphere S^{d-1}.
    
    Args:
        f_score: Function x -> scalar score (to maximize)
        f_grad: Function x -> gradient vector
        d: Dimension
        lr: Learning rate
        max_steps: Maximum iterations
        patience: Early stopping patience
        tol: Convergence tolerance
        seed: Random seed for initialization
        
    Returns:
        x_star: Optimized vector on sphere
        info: Dict with convergence info
    """
    # Set seed and initialize
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.randn(d)
    x = x / np.linalg.norm(x)  # Project to sphere
    
    best_score = f_score(x)
    best_x = x.copy()
    counter = 0
    scores = [best_score]
    
    for step in range(max_steps):
        # Compute Euclidean gradient
        g = f_grad(x)
        
        # Project to tangent space (Riemannian gradient)
        g_R = g - np.dot(x, g) * x
        
        # Gradient ascent step
        x = x + lr * g_R
        
        # Retract to sphere
        x = x / np.linalg.norm(x)
        
        # Evaluate
        score = f_score(x)
        scores.append(score)
        
        # Early stopping
        if score > best_score + tol:
            best_score = score
            best_x = x.copy()
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            break
    
    info = {
        "final_score": best_score,
        "n_iterations": step + 1,
        "converged": counter >= patience,
        "scores": scores,
    }
    
    return best_x, info


def create_and_operator(v_A: np.ndarray, v_B: np.ndarray, v_AB: np.ndarray):
    """
    Create AND operator scoring function and gradient.
    
    Score: 2 * S_AB - max(S_A, S_B)
    where S_X = dot(x, v_X)
    """
    def f_score(x):
        S_A = np.dot(x, v_A)
        S_B = np.dot(x, v_B)
        S_AB = np.dot(x, v_AB)
        return 2 * S_AB - max(S_A, S_B)
    
    def f_grad(x):
        S_A = np.dot(x, v_A)
        S_B = np.dot(x, v_B)
        # Gradient of max(S_A, S_B) is v_A or v_B depending on which is larger
        if S_A > S_B:
            return 2 * v_AB - v_A
        else:
            return 2 * v_AB - v_B
    
    return f_score, f_grad


def create_not_operator(v_A: np.ndarray, v_B: np.ndarray, v_AB: np.ndarray):
    """
    Create NOT operator scoring function and gradient.
    
    Score: 2 * S_A - S_AB
    """
    def f_score(x):
        S_A = np.dot(x, v_A)
        S_AB = np.dot(x, v_AB)
        return 2 * S_A - S_AB
    
    def f_grad(x):
        return 2 * v_A - v_AB
    
    return f_score, f_grad


def run_seed_variance_analysis(
    n_queries: int = 100,
    n_seeds: int = 5,
    d: int = 768,
    operator: str = "AND",
):
    """
    Run seed variance analysis with synthetic query vectors.
    
    Args:
        n_queries: Number of test queries
        n_seeds: Number of random seeds per query
        d: Embedding dimension
        operator: "AND" or "NOT"
    """
    print(f"Running SQO seed variance analysis")
    print(f"  Queries: {n_queries}")
    print(f"  Seeds per query: {n_seeds}")
    print(f"  Dimension: {d}")
    print(f"  Operator: {operator}")
    print()
    
    # Store results
    all_variances = []
    all_score_diffs = []
    all_times = []
    
    for q in range(n_queries):
        if (q + 1) % 20 == 0:
            print(f"  Processing query {q + 1}/{n_queries}")
        
        # Generate random query vectors (normalized)
        v_A = np.random.randn(d)
        v_A = v_A / np.linalg.norm(v_A)
        
        v_B = np.random.randn(d)
        v_B = v_B / np.linalg.norm(v_B)
        
        v_AB = np.random.randn(d)
        v_AB = v_AB / np.linalg.norm(v_AB)
        
        # Create operator
        if operator == "AND":
            f_score, f_grad = create_and_operator(v_A, v_B, v_AB)
        else:
            f_score, f_grad = create_not_operator(v_A, v_B, v_AB)
        
        # Run SQO with different seeds
        seed_scores = []
        seed_times = []
        seed_x_stars = []
        
        for s in range(n_seeds):
            start_time = time.time()
            x_star, info = riemannian_gradient_descent(
                f_score, f_grad, d,
                seed=q * 1000 + s,  # Deterministic seed per query+seed combo
            )
            elapsed = time.time() - start_time
            
            seed_scores.append(info["final_score"])
            seed_times.append(elapsed)
            seed_x_stars.append(x_star)
        
        # Compute variance metrics
        score_variance = np.var(seed_scores)
        score_std = np.std(seed_scores)
        score_range = max(seed_scores) - min(seed_scores)
        mean_time = np.mean(seed_times)
        
        all_variances.append(score_variance)
        all_score_diffs.append(score_range)
        all_times.append(mean_time)
    
    # Aggregate statistics
    results = {
        "n_queries": n_queries,
        "n_seeds": n_seeds,
        "dimension": d,
        "operator": operator,
        "score_variance": {
            "mean": float(np.mean(all_variances)),
            "std": float(np.std(all_variances)),
            "max": float(np.max(all_variances)),
        },
        "score_range_across_seeds": {
            "mean": float(np.mean(all_score_diffs)),
            "std": float(np.std(all_score_diffs)),
            "max": float(np.max(all_score_diffs)),
        },
        "time_per_query_seconds": {
            "mean": float(np.mean(all_times)),
            "std": float(np.std(all_times)),
        },
    }
    
    # Print results
    print("\n" + "="*60)
    print("SQO Seed Variance Analysis Results")
    print("="*60)
    print(f"Operator: {operator}")
    print(f"Queries evaluated: {n_queries}")
    print(f"Seeds per query: {n_seeds}")
    print()
    print("Score Variance (across seeds within each query):")
    print(f"  Mean variance: {results['score_variance']['mean']:.8f}")
    print(f"  Std of variance: {results['score_variance']['std']:.8f}")
    print(f"  Max variance: {results['score_variance']['max']:.8f}")
    print()
    print("Score Range (max - min across seeds):")
    print(f"  Mean range: {results['score_range_across_seeds']['mean']:.6f}")
    print(f"  Max range: {results['score_range_across_seeds']['max']:.6f}")
    print()
    print("Timing:")
    print(f"  Mean time per query: {results['time_per_query_seconds']['mean']*1000:.2f} ms")
    print("="*60)
    
    # Interpretation
    if results['score_range_across_seeds']['mean'] < 0.001:
        print("\n✓ CONCLUSION: Seed variance is NEGLIGIBLE (<0.1% score range)")
        print("  SQO converges to effectively identical optima regardless of initialization.")
    elif results['score_range_across_seeds']['mean'] < 0.01:
        print("\n✓ CONCLUSION: Seed variance is LOW (<1% score range)")
        print("  Minor initialization effects, but results are stable.")
    else:
        print("\n⚠ CONCLUSION: Seed variance may be significant")
        print("  Consider averaging across seeds or using deterministic initialization.")
    
    return results


def run_with_real_embeddings(
    model_name: str,
    queries_path: str,
    n_queries: int = 100,
    n_seeds: int = 5,
):
    """
    Run seed variance analysis with real embeddings from your encoder.
    
    This function should be adapted to your specific embedding setup.
    """
    # TODO: Implement with your actual embedding loading code
    # This would load real query embeddings and run SQO on them
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQO Seed Variance Analysis")
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--operator", type=str, default="AND", choices=["AND", "NOT"])
    parser.add_argument("--output", type=str, default="sqo_seed_variance.json")
    args = parser.parse_args()
    
    results = run_seed_variance_analysis(
        n_queries=args.n_queries,
        n_seeds=args.n_seeds,
        d=args.dimension,
        operator=args.operator,
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
