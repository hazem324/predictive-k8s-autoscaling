import requests

PROMETHEUS = "http://localhost:9090"

def query(q):
    try:
        r = requests.get(
            f"{PROMETHEUS}/api/v1/query",
            params={"query": q},
            timeout=5
        )

        res = r.json()["data"]["result"]
        value = float(res[0]["value"][1]) if res else 0.0

        return value

    except Exception as e:
        print("Error:", e)
        return 0.0


metrics = {
    "cpu_mean": 'avg(rate(container_cpu_usage_seconds_total{namespace="default"}[1m]))*100',

    "cpu_max": 'max(rate(container_cpu_usage_seconds_total{namespace="default"}[1m]))*100',

    "mem_mean": 'avg(container_memory_usage_bytes{namespace="default"}/container_spec_memory_limit_bytes)*100',

    "mem_max": 'max(container_memory_usage_bytes{namespace="default"}/container_spec_memory_limit_bytes)*100',

    "wep": 'sum(rate(http_requests_total{namespace="default"}[1m]))*60'
}

for name, promql in metrics.items():
    print(f"\n{name}")
    print(f"Query: {promql}")

    value = query(promql)

    print(f"Value: {value}")