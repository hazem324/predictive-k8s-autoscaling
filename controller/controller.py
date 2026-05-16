from kubernetes import client, config
import pickle, time
from prometheus_client import get_metrics
from features import compute_temporal

# ── Connect to Kubernetes from inside the pod ──────────────────
config.load_incluster_config()
apps_v1 = client.AppsV1Api()

# ── Load the trained model ─────────────────────────────────────
with open("/app/model.pkl", "rb") as f:
    model = pickle.load(f)

# ── Feature order must match Colab training EXACTLY ───────────
FEATURES = [
    "cpu_mean","cpu_max","mem_mean","mem_max","wep",
    "priority_mean","assigned_mem","page_cache",
    "hour","day_of_week","is_weekend",
    "cpu_lag1","cpu_lag3","cpu_lag5","wep_lag1","wep_lag3",
    "cpu_roll5_mean","cpu_roll5_max","cpu_roll5_std","wep_roll5_mean",
    "cpu_trend","wep_trend","spike_persist","cpu_accel"
]

DEPLOYMENT = "my-app"
NAMESPACE  = "default"
history    = []   # rolling buffer — last 6 minutes of metrics

def scale(n):
    patch = {"spec": {"replicas": n}}
    apps_v1.patch_namespaced_deployment_scale(DEPLOYMENT, NAMESPACE, patch)

def get_replicas():
    return apps_v1.read_namespaced_deployment(DEPLOYMENT, NAMESPACE).spec.replicas or 1

print(" AI Controller started — model loaded")
while True:
    try:
        # 1. Read live metrics from Prometheus
        metrics = get_metrics()

        # 2. Add temporal features (lag, rolling, trend)
        metrics = compute_temporal(metrics, history)

        # 3. Build feature vector in correct order
        features = [[metrics[f] for f in FEATURES]]

        # 4. Ask the model for the scaling decision
        decision = model.predict(features)[0]
        current  = get_replicas()

        print(f"cpu={metrics['cpu_mean']:.2f}%  "
              f"trend={metrics['cpu_trend']:.3f}  "
              f"spike={metrics['spike_persist']}  "
              f"→ {decision}  (pods={current})")

        # 5. Apply the decision
        if decision == "scale_up":
            scale(min(10, current + 2))
            print(f"  ↑ {current} → {min(10, current+2)} pods")
        elif decision == "scale_down":
            scale(max(1, current - 1))
            print(f"  ↓ {current} → {max(1, current-1)} pods")
        else:
            print(f"  → No change ({current} pods)")

        # 6. Update history buffer (keep last 6 minutes)
        history.append(metrics)
        if len(history) > 6:
            history.pop(0)

    except Exception as e:
        print(f" Error: {e}")

    time.sleep(60)   # wait 1 minute before next cycle