import requests
from datetime import datetime

PROMETHEUS = "http://monitoring-kube-prometheus-prometheus:9090"


def query(q: str) -> float:
    try:
        r = requests.get(
            f"{PROMETHEUS}/api/v1/query",
            params={"query": q},
            timeout=5,
        )
        res = r.json()["data"]["result"]
        return float(res[0]["value"][1]) if res else 0.0
    except Exception:
        return 0.0


def get_metrics(deployment: str, namespace: str) -> dict:
    """
    Fetch live metrics scoped to a specific deployment inside a namespace.

    Pods belonging to a Deployment are matched by the label  app=<deployment>
    which is the default label set by `kubectl create deployment` and most Helm
    charts.  If your pods use a different label key, change the selector below.
    """
    now = datetime.now()

    # Label selector shared by every query for this deployment
    sel = f'namespace="{namespace}", pod=~"^{deployment}-.*"'

    return {
        # ── CPU (%) ───────────────────────────────────────────────────────────
        "cpu_mean": query(
            f'avg(rate(container_cpu_usage_seconds_total{{{sel}}}[1m])) * 100'
        ),
        "cpu_max": query(
            f'max(rate(container_cpu_usage_seconds_total{{{sel}}}[1m])) * 100'
        ),

        # ── Memory (% of limit) ───────────────────────────────────────────────
        "mem_mean": query(
    f'''
    avg(
      container_memory_usage_bytes{{{sel}}}
    )
    /
    avg(
      kube_pod_container_resource_limits{{
        namespace="{namespace}",
        pod=~"^{deployment}-.*",
        resource="memory"
      }}
    ) * 100
    '''
),

"mem_max": query(
    f'''
    max(
      container_memory_usage_bytes{{{sel}}}
    )
    /
    max(
      kube_pod_container_resource_limits{{
        namespace="{namespace}",
        pod=~"^{deployment}-.*",
        resource="memory"
      }}
    ) * 100
    '''
),

        # ── Requests per minute ───────────────────────────────────────────────
        "wep": query(
    f'sum(rate(nginx_ingress_controller_requests{{exported_namespace="{namespace}"}}[1m])) * 60'
),

        # ── Borg-specific features not available in Prometheus ────────────────
        "priority_mean": 100.0,
        "assigned_mem":  0.01,
        "page_cache":    0.005,

        # ── Time features (computed locally, same for every deployment) ───────
        "hour":        now.hour,
        "day_of_week": now.weekday(),
        "is_weekend":  int(now.weekday() >= 5),
    }