# collect_metrics.py
# Run this locally while your task-flow app is running in Minikube
# Let it run for at least 1-2 hours to collect enough data

import requests
import pandas as pd
import time
from datetime import datetime

PROMETHEUS = "http://localhost:9090"  # minikube: kubectl port-forward svc/prometheus 9090:9090
NAMESPACE = "task-flow"               # your namespace
INTERVAL   = 60                       # collect one row every 60 seconds

def query(q: str) -> float:
    try:
        r = requests.get(f"{PROMETHEUS}/api/v1/query", params={"query": q}, timeout=5)
        res = r.json()["data"]["result"]
        return float(res[0]["value"][1]) if res else 0.0
    except:
        return 0.0

def collect_row(namespace):
    now = datetime.now()

    sel = f'namespace="{namespace}"'

    return {
        "timestamp": now.isoformat(),

        # CPU across all pods in namespace
        "cpu_mean": query(
            f'avg(rate(container_cpu_usage_seconds_total{{{sel}}}[1m])) * 100'
        ),

        "cpu_max": query(
            f'max(rate(container_cpu_usage_seconds_total{{{sel}}}[1m])) * 100'
        ),

        # Memory across all pods in namespace
        "mem_mean": query(
            f'avg(container_memory_usage_bytes{{{sel}}})'
        ) / (1024 * 1024),

        "mem_max": query(
            f'max(container_memory_usage_bytes{{{sel}}})'
        ) / (1024 * 1024),

        # Total pod count
        "pods": query(
            f'count(kube_pod_info{{namespace="{namespace}"}})'
        ),

        # Total requests per minute
        "wep": query(
            f'sum(rate(nginx_ingress_controller_requests{{exported_namespace="{namespace}"}}[1m])) * 60'
        ),

        "hour": now.hour,
        "day_of_week": now.weekday(),
        "is_weekend": int(now.weekday() >= 5),
    }

# ── Main collection loop ──────────────────────────────────────────────
rows = []
print(f"Collecting metrics every {INTERVAL}s — press Ctrl+C to stop and save")
print(f" namespace {NAMESPACE}\n")

try:
    while True:
        row = collect_row(NAMESPACE)
        rows.append(row)
        print(f"[{row['timestamp']}] cpu={row['cpu_mean']:.2f}%  mem={row['mem_mean']:.2f}MB  pods={row['pods']:.0f}  req={row['wep']:.1f}  wep={row['wep']:.1f}")
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    df = pd.DataFrame(rows)
    filename = f"minikube_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nSaved {len(df)} rows to {filename}")
    print(df.describe().round(2))