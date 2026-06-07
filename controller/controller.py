from kubernetes import client, config
import pickle, time, threading
from prometheus_client import get_metrics
from features import compute_temporal

# ── Connect to Kubernetes from inside the pod ────────────────────────────────
config.load_incluster_config()
apps_v1    = client.AppsV1Api()
core_v1    = client.CoreV1Api()

# ── Detect the namespace this controller is running in ──────────────────────
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
print(f"Controller namespace: {NAMESPACE}")

# ── Load the trained model ───────────────────────────────────────────────────
with open("/app/model.pkl", "rb") as f:
    model = pickle.load(f)

# ── Feature order must match Colab training EXACTLY ─────────────────────────
FEATURES = [
    "cpu_mean", "cpu_max", "mem_mean", "mem_max", "wep",
    "priority_mean", "assigned_mem", "page_cache",
    "hour", "day_of_week", "is_weekend",
    "cpu_lag1", "cpu_lag3", "cpu_lag5", "wep_lag1", "wep_lag3",
    "cpu_roll5_mean", "cpu_roll5_max", "cpu_roll5_std", "wep_roll5_mean",
    "cpu_trend", "wep_trend", "spike_persist", "cpu_accel"
]

SCALE_INTERVAL   = 60   # seconds between each decision cycle
HISTORY_MAX_LEN  = 6    # rolling buffer length (6 minutes)
MAX_REPLICAS     = 10
MIN_REPLICAS     = 1


# ── Per-deployment helpers ───────────────────────────────────────────────────
def scale(deployment: str, n: int) -> None:
    patch = {"spec": {"replicas": n}}
    apps_v1.patch_namespaced_deployment_scale(deployment, NAMESPACE, patch)


def get_replicas(deployment: str) -> int:
    return (
        apps_v1.read_namespaced_deployment(deployment, NAMESPACE).spec.replicas or 1
    )


def list_deployments() -> list[str]:
    """Return names of all Deployments currently in NAMESPACE."""
    deploys = apps_v1.list_namespaced_deployment(NAMESPACE)
    return [d.metadata.name for d in deploys.items]


# ── Independent scaling loop for a single Deployment ────────────────────────
def deployment_loop(deployment: str) -> None:
    """Runs forever in its own thread; maintains its own history buffer."""
    history: list[dict] = []
    print(f"[{deployment}] Scaling loop started")

    while True:
        try:
            # 1. Read live metrics from Prometheus (scoped to this deployment/pods)
            metrics = get_metrics(deployment=deployment, namespace=NAMESPACE)

            # 2. Add temporal features (lag, rolling, trend)
            metrics = compute_temporal(metrics, history)

            # 3. Build feature vector in correct order
            features = [[metrics[f] for f in FEATURES]]

            # 4. Ask the model for the scaling decision
            decision = model.predict(features)[0]
            current  = get_replicas(deployment)

            print(
                f"[{deployment}] "
                f"cpu={metrics['cpu_mean']:.2f}%  "
                f"trend={metrics['cpu_trend']:.3f}  "
                f"spike={metrics['spike_persist']}  "
                f"→ {decision}  (pods={current})"
            )

            # 5. Apply the decision
            if decision == "scale_up":
                new = min(MAX_REPLICAS, current + 2)
                scale(deployment, new)
                print(f"[{deployment}]  ↑ {current} → {new} pods")

            elif decision == "scale_down":
                new = max(MIN_REPLICAS, current - 1)
                scale(deployment, new)
                print(f"[{deployment}]  ↓ {current} → {new} pods")

            else:
                print(f"[{deployment}]  → No change ({current} pods)")

            # 6. Update history buffer (keep last HISTORY_MAX_LEN minutes)
            history.append(metrics)
            if len(history) > HISTORY_MAX_LEN:
                history.pop(0)

        except Exception as e:
            print(f"[{deployment}] Error: {e}")

        time.sleep(SCALE_INTERVAL)


# ── Watcher: discovers new Deployments and spawns threads for them ───────────
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
                    daemon=True,          # dies automatically when main thread exits
                )
                t.start()
                active.add(name)
                print(f"[watcher] New deployment detected — started loop for '{name}'")

            # Log deployments that disappeared (threads will die on next API call)
            for name in active - current_deploys:
                print(f"[watcher] Deployment '{name}' no longer found — loop will exit")
                active.discard(name)

        except Exception as e:
            print(f"[watcher] Error listing deployments: {e}")

        time.sleep(30)   # re-check for new/removed deployments every 30 s


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"AI Controller started — namespace={NAMESPACE}  model loaded")
    watcher_loop()   # runs forever in the main thread