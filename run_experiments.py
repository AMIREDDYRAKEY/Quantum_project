"""
Benchmark runner: Evaluates Classical KNN, Z-QKNN, ZZ-QKNN, Pauli-QKNN, Hybrid-QKNN,
and Adaptive Hybrid QKNN across all 4 datasets.
Saves results to JSON and CSV.
"""
import json
import pandas as pd
from backend.core.data_manager import DataManager
from backend.core.adaptive_hybrid_qknn import run_full_benchmark_comparison

dm = DataManager(random_state=42)
all_benchmarks = {}

for ds_name in ["iris", "wine", "rain", "indian_liver"]:
    data = dm.split_and_preprocess(ds_name, n_features=4 if ds_name != "iris" else None)
    print(f"Benchmarking {ds_name.upper()} (Train={data['n_train']}, Test={data['n_test']})...")
    res = run_full_benchmark_comparison(data, k=3, candidate_k=[1, 3, 5], n_wires=4)
    all_benchmarks[ds_name] = res

# Save to JSON
with open("backend/data/benchmark_results/live_experiment_results.json", "w") as f:
    json.dump(all_benchmarks, f, indent=2)

print("\n" + "=" * 90)
print(f"{'DATASET':<14} | {'MODEL':<26} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8}")
print("=" * 90)

table_rows = []
for ds, res in all_benchmarks.items():
    for model_key in ["classical_knn", "z_qknn", "zz_qknn", "pauli_qknn", "hybrid_qknn", "adaptive_hybrid_qknn"]:
        m = res[model_key]
        name = m.get("model_type", model_key)
        acc = m["accuracy"]
        prec = m["precision"]
        rec = m["recall"]
        f1 = m["f1_score"]
        print(f"{ds:<14} | {name:<26} | {acc:<8.4f} | {prec:<8.4f} | {rec:<8.4f} | {f1:<8.4f}")
        table_rows.append({
            "dataset": ds,
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
    sum_info = res["summary_analysis"]
    print(f"  --> AH-QKNN Selected: Map={sum_info['selected_optimal_map']}, K={sum_info['selected_optimal_k']}, F1-Delta vs Hybrid: {sum_info['f1_improvement']:+.4f}")
    print("-" * 90)

pd.DataFrame(table_rows).to_csv("backend/data/benchmark_results/live_experiment_results.csv", index=False)
print("\nAll experiment results saved to backend/data/benchmark_results/live_experiment_results.json and .csv")
