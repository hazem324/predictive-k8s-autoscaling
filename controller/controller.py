from kubernetes import client, config
import pickle, time, threading
from prometheus_client import get_metrics
from features import compute_temporal
import pandas as pd


# Connect to Kubernetes from inside the pod
try:
    config.load_incluster_config()
    print("Running inside Kubernetes", flush=True)
except Exception:
    config.load_kube_config()
    print("Running locally")
apps_v1    = client.AppsV1Api()
core_v1    = client.CoreV1Api()

# Detect the namespace this controller is running in
def get_current_namespace() -> str:
    """Read namespace from the service-account token mounted in every pod."""
    ns_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    try:
        with open(ns_file) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"Cannot read namespace from {ns_file}. "
            "Make sure the controller is running inside a Kubernetes pod."
        )

NAMESPACE = get_current_namespace()
print(f"Controller namespace: {NAMESPACE}", flush=True)

# Load the trained model
with open("/app/model/model.pkl", "rb") as f:
    model = pickle.load(f)
    print(model.feature_names_in_, flush=True)

# Feature order must match Colab training EXACTLY
FEATURES = [
    "cpu_mean", "cpu_max", "mem_mean", "mem_max", "wep",
    "priority_mean", "assigned_mem", "page_cache",
    "hour", "day_of_week", "is_weekend",
    "cpu_lag1", "cpu_lag3", "cpu_lag5", "wep_lag1", "wep_lag3",
    "cpu_roll5_mean", "cpu_roll5_max", "cpu_roll5_std", "wep_roll5_mean",
    "cpu_trend", "wep_trend", "spike_persist", "cpu_accel"
]

SCALE_INTERVAL   = 40   # seconds between each decision cycle
HISTORY_MAX_LEN  = 6    # rolling buffer length (6 minutes)
MAX_REPLICAS     = 10
MIN_REPLICAS     = 1

# ── AI Controller thresholds ──────────────────────────────────────────
# The model predicts future CPU (a number). These thresholds convert
# that prediction into a scaling action. Change these without retraining.
SCALE_UP_THRESHOLD   = 85   # predicted CPU above this → scale up
SCALE_DOWN_THRESHOLD = 30   # predicted CPU below this → scale down

def ai_controller_decision(predicted_cpu: float) -> str:
    """Convert a predicted future CPU value into a scaling action."""
    if predicted_cpu > SCALE_UP_THRESHOLD:
        return "scale_up"
    elif predicted_cpu < SCALE_DOWN_THRESHOLD:
        return "scale_down"
    else:
        return "keep"


# Per-deployment helpers
def scale(deployment: str, n: int) -> None:
    patch = {"spec": {"replicas": n}}
    apps_v1.patch_namespaced_deployment_scale(deployment, NAMESPACE, patch)


def get_replicas(deployment: str) -> int:
    return (
        apps_v1.read_namespaced_deployment(deployment, NAMESPACE).spec.replicas or 1
    )


def list_deployments() -> list[str]:
    """Return names of all Deployments currently in NAMESPACE."""
    deploys = apps_v1.list_namespaced_deployment(NAMESPACE, label_selector="ai-scaling=enabled")
    return [d.metadata.name for d in deploys.items]


# Independent scaling loop for a single Deployment
def deployment_loop(deployment: str) -> None:
    """Runs forever in its own thread; maintains its own history buffer."""
    history: list[dict] = []
    print(f"[{deployment}] Scaling loop started", flush=True)

    while True:
        try:
            # 1. Read live metrics from Prometheus (scoped to this deployment/pods)
            metrics = get_metrics(deployment=deployment, namespace=NAMESPACE)

            print(f"[{deployment}] RAW METRICS = {metrics}", flush=True)

            # 2. Add temporal features (lag, rolling, trend)
            metrics = compute_temporal(metrics, history)

            # 3. Build feature vector in correct order
            features = pd.DataFrame(
                [[metrics[f] for f in FEATURES]],
                columns=FEATURES
            )

            # 4. Ask the model for the predicted future CPU (a number, not a label)
            predicted_cpu = model.predict(features)[0]

            # 5. AI Controller converts the prediction into a scaling decision
            decision = ai_controller_decision(predicted_cpu)
            current  = get_replicas(deployment)

            print("\n" + "=" * 60)
            print(
                f"[{deployment}] "
                f"cpu={metrics['cpu_mean']:.2f}%  "
                f"trend={metrics['cpu_trend']:.3f}  "
                f"spike={metrics['spike_persist']}  "
                f"→ predicted_cpu={predicted_cpu:.1f}%  "
                f"→ {decision}  (pods={current})", flush=True
            )

            # 6. Apply the decision
            if decision == "scale_up":
                new = min(MAX_REPLICAS, current + 2)
                scale(deployment, new)
                print(f"[{deployment}] SCALE UP   {current} -> {new}", flush=True)

            elif decision == "scale_down":
                new = max(MIN_REPLICAS, current - 1)
                scale(deployment, new)
                print(f"[{deployment}] SCALE DOWN {current} -> {new}", flush=True)

            else:
                print(f"[{deployment}] NO CHANGE  {current}", flush=True)

            # 7. Update history buffer (keep last HISTORY_MAX_LEN minutes)
            history.append(metrics)
            if len(history) > HISTORY_MAX_LEN:
                history.pop(0)

        except Exception as e:
            print(f"[{deployment}] Error: {e}")

        time.sleep(SCALE_INTERVAL)


# Watcher: discovers new Deployments and spawns threads for them
def watcher_loop() -> None:
    """
    Polls for new Deployments every 30 s.
    Spawns a daemon thread for each one not yet tracked.
    Threads for deleted Deployments will self-exit on the next RBAC/API error.
    """
    active: set[str] = set()

    while True:
        try:
            current_deploys = set(list_deployments())

            # Start a thread for every new Deployment
            for name in current_deploys - active:
                t = threading.Thread(
                    target=deployment_loop,
                    args=(name,),
                    name=f"scaler-{name}",
                    daemon=True,
                )
                t.start()
                active.add(name)
                print(f"[watcher] New deployment detected — started loop for '{name}'")

            # Log deployments that disappeared
            for name in active - current_deploys:
                print(f"[watcher] Deployment '{name}' no longer found — loop will exit")
                active.discard(name)

        except Exception as e:
            print(f"[watcher] Error listing deployments: {e}")

        time.sleep(30)


# Entry point
if __name__ == "__main__":
    print(f"AI Controller started — namespace={NAMESPACE}  model loaded")
    watcher_loop()