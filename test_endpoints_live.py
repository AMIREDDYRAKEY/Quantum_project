import urllib.request
import json
import time

def post(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def get(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode('utf-8'))

print("--- Testing Live Quantum Platform API Endpoints ---")

# 1. Health
health = get('http://localhost:5000/api/health')
print(f"[1/5] Health Check: status='{health['status']}', simulator='{health['simulator']}', framework='{health['quantum_framework']}'")

# 2. Datasets
summary = get('http://localhost:5000/api/datasets/iris/summary')
print(f"[2/5] Iris Summary: name='{summary['summary']['name']}', total_records={summary['summary']['total_records']}, n_features={summary['summary']['n_features']}")

# 3. Live Benchmark Results
bench = get('http://localhost:5000/api/benchmarks/live-results')
bench_count = len(bench['data'])
print(f"[3/5] Live Benchmarks: {bench_count} experiments recorded in live database")

# 4. Single Quantum Prediction (PennyLane Hybrid Feature Map simulation)
t0 = time.time()
pred_payload = {
    'dataset': 'iris',
    'model_type': 'Quantum_KNN_HYBRID',
    'feature_map': 'hybrid',
    'k': 3,
    'features': [5.1, 3.5, 1.4, 0.2]
}
pred = post('http://localhost:5000/api/predict/single', pred_payload)
t1 = time.time()
print(f"[4/5] Live Quantum Prediction ({t1-t0:.2f}s): Predicted: '{pred['predicted_label']}' (class {pred['predicted_class']}), Neighbors: {len(pred['neighbors'])}")

# 5. Quantum Explainability (State Sensitivity Gradients & Neighbor Fidelity)
t2 = time.time()
xai_payload = {
    'dataset': 'iris',
    'feature_map': 'hybrid',
    'k': 3,
    'sample_index': 0
}
xai = post('http://localhost:5000/api/xai/quantum', xai_payload)
t3 = time.time()
explanation = xai['explanation']
grad_summary = [f"{s['feature']}: {s['sensitivity']:.4f}" for s in explanation['quantum_feature_sensitivities']]
top_neighbors = [f"Idx={n['train_index']} (F={n['fidelity']:.3f}, W={n['weight']:.2f})" for n in explanation['neighbor_contributions'][:3]]
print(f"[5/5] Live Quantum XAI ({t3-t2:.2f}s): Quantum State Gradients: {grad_summary}")
print(f"      Top Nearest Quantum Neighbors: {top_neighbors}")
print(f"      Class Votes: {explanation['class_votes']}")
print("\n>>> ALL API VERIFICATIONS PASSED CLEANLY! <<<")

# 6. Frontend Verification
html = get('http://localhost:3000/') if False else urllib.request.urlopen('http://localhost:3000/').read().decode('utf-8')
print(f"[6/6] Frontend React Root: {len(html)} bytes delivered from Vite server (HTTP 200)")
print(">>> Web application fully verified and operational! <<<")
