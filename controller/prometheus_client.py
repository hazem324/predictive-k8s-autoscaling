import requests
from datetime import datetime

PROMETHEUS = "http://monitoring-kube-prometheus-prometheus:9090"

def query(q):
    try:
        r = requests.get(f"{PROMETHEUS}/api/v1/query",
                         params={"query": q}, timeout=5)
        res = r.json()["data"]["result"]
        return float(res[0]["value"][1]) if res else 0.0
    except:
        return 0.0

def get_metrics():
    now = datetime.now()
    return {
        # Current metrics — from Prometheus
        "cpu_mean": query('avg(rate(container_cpu_usage_seconds_total{namespace="default"}[1m]))*100'),
        "cpu_max":  query('max(rate(container_cpu_usage_seconds_total{namespace="default"}[1m]))*100'),
        "mem_mean": query('avg(container_memory_usage_bytes{namespace="default"}/container_spec_memory_limit_bytes)*100'),
        "mem_max":  query('max(container_memory_usage_bytes{namespace="default"}/container_spec_memory_limit_bytes)*100'),
        "wep":      query('sum(rate(http_requests_total{namespace="default"}[1m]))*60'),
        # Defaults for Borg-specific features not available in Prometheus
        "priority_mean": 100.0,
        "assigned_mem":  0.01,
        "page_cache":    0.005,
        # Time features — computed locally
        "hour":        now.hour,
        "day_of_week": now.weekday(),
        "is_weekend":  int(now.weekday() >= 5),
    }