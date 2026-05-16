"""
=============================================================
  load_generator.py
  Test the AI autoscaling model LOCALLY before deploying to Kubernetes
=============================================================

HOW TO USE:
  1. Run this file: python load_generator.py
  2. Choose a scenario from the menu
  3. Watch the model predict: scale_up / keep / scale_down

NO Kubernetes needed. NO Docker needed. NO Prometheus needed.
Only model.pkl is required.

PREREQUISITE:
  pip install scikit-learn numpy requests

FILE LOCATION:
  ai-autoscaling/scripts/load_generator.py
  ai-autoscaling/model/model.pkl          ← must exist

=============================================================
"""

import pickle
import numpy as np
import time
import sys
import os

# ── Load the model ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ model.pkl loaded successfully")
    print(f"   Features expected : {model.n_features_in_}")
    print(f"   Classes           : {list(model.classes_)}")
except FileNotFoundError:
    print(f"❌ model.pkl not found at: {MODEL_PATH}")
    print(f"   Make sure model.pkl is in ai-autoscaling/model/model.pkl")
    sys.exit(1)


# ── Feature order — MUST match Colab training exactly ──────────
FEATURES = [
    "cpu_mean",       # average CPU this minute (%)
    "cpu_max",        # peak CPU this minute (%)
    "mem_mean",       # average memory (%)
    "mem_max",        # peak memory (%)
    "wep",            # workload events per minute
    "priority_mean",  # average job priority
    "assigned_mem",   # reserved memory
    "page_cache",     # OS page cache
    "hour",           # hour of day (0-23)
    "day_of_week",    # 0=Monday, 6=Sunday
    "is_weekend",     # 0 or 1
    "cpu_lag1",       # CPU 1 min ago
    "cpu_lag3",       # CPU 3 min ago
    "cpu_lag5",       # CPU 5 min ago
    "wep_lag1",       # workload 1 min ago
    "wep_lag3",       # workload 3 min ago
    "cpu_roll5_mean", # rolling 5-min avg CPU
    "cpu_roll5_max",  # rolling 5-min max CPU
    "cpu_roll5_std",  # rolling 5-min CPU volatility
    "wep_roll5_mean", # rolling 5-min avg workload
    "cpu_trend",      # CPU change over last 3 min
    "wep_trend",      # workload change over last 3 min
    "spike_persist",  # 1 if spike sustained 3+ min
    "cpu_accel",      # acceleration of trend
]


# ── Helper: compute temporal features from history ─────────────
def compute_temporal(current, history):
    """
    Recompute lag, rolling, trend, spike from history buffer.
    This replicates features.py — same logic used in controller.py
    """
    cpu = [h["cpu_mean"] for h in history] + [current["cpu_mean"]]
    wep = [h["wep"]      for h in history] + [current["wep"]]

    def safe_get(series, n):
        return series[-n] if len(series) >= n else series[0]

    current["cpu_lag1"] = safe_get(cpu, 2)
    current["cpu_lag3"] = safe_get(cpu, 4)
    current["cpu_lag5"] = safe_get(cpu, 6)
    current["wep_lag1"] = safe_get(wep, 2)
    current["wep_lag3"] = safe_get(wep, 4)

    last5 = cpu[-5:]
    current["cpu_roll5_mean"] = float(np.mean(last5))
    current["cpu_roll5_max"]  = float(np.max(last5))
    current["cpu_roll5_std"]  = float(np.std(last5))
    current["wep_roll5_mean"] = float(np.mean(wep[-5:]))

    current["cpu_trend"] = cpu[-1] - safe_get(cpu, 4)
    current["wep_trend"] = wep[-1] - safe_get(wep, 4)

    current["spike_persist"] = int(
        current["cpu_max"] > current["cpu_roll5_mean"] * 2
    )

    prev_trend = (cpu[-2] - safe_get(cpu, 5)) if len(cpu) >= 5 else 0
    current["cpu_accel"] = current["cpu_trend"] - prev_trend

    return current


# ── Helper: predict from raw metrics ───────────────────────────
def predict(metrics, history):
    metrics  = compute_temporal(metrics, history)
    features = [[metrics[f] for f in FEATURES]]
    decision = model.predict(features)[0]
    probas   = model.predict_proba(features)[0]
    classes  = list(model.classes_)
    conf     = max(probas) * 100
    proba_dict = dict(zip(classes, probas))
    return decision, conf, proba_dict, metrics


# ── Helper: print one prediction result ────────────────────────
def print_result(decision, conf, proba_dict, metrics, minute=None):
    icons = {"scale_up": "↑  SCALE UP  ", "keep": "→  KEEP      ", "scale_down": "↓  SCALE DOWN"}
    colors = {"scale_up": "\033[91m", "keep": "\033[93m", "scale_down": "\033[92m"}
    reset  = "\033[0m"

    prefix = f"  Minute {minute:2d}" if minute is not None else "  Result   "
    icon   = icons.get(decision, "?")
    color  = colors.get(decision, "")

    print(f"{prefix} | cpu={metrics['cpu_mean']:5.2f}%  "
          f"trend={metrics['cpu_trend']:+6.3f}  "
          f"spike={int(metrics['spike_persist'])}  "
          f"| {color}{icon}{reset}  "
          f"confidence={conf:.1f}%")

    # Probability bar for each class
    for cls in sorted(proba_dict.keys()):
        prob = proba_dict[cls]
        bar  = "█" * int(prob * 25)
        print(f"           {cls:12s}: {bar:<25s} {prob*100:.1f}%")
    print()


# ══════════════════════════════════════════════════════════════
#  SCENARIO 1 — Single instant prediction
# ══════════════════════════════════════════════════════════════
def scenario_single():
    """
    Predict from one set of hand-crafted metrics.
    Useful to test one specific situation.
    """
    print("\n" + "═" * 60)
    print("  SCENARIO 1 — Single Instant Prediction")
    print("═" * 60)
    print("  Edit the values below to simulate any situation.\n")

    # ── Modify these values to test different situations ──────
    metrics_now = {
        "cpu_mean":      3.5,    # % — try 0.1 (quiet) or 8.0 (overload)
        "cpu_max":       8.2,    # % — peak CPU this minute
        "mem_mean":      1.2,    # % — memory usage
        "mem_max":       2.8,    # % — peak memory
        "wep":           80,     # events/min — workload intensity
        "priority_mean": 150.0,  # job priority
        "assigned_mem":  0.01,
        "page_cache":    0.005,
        "hour":          10,     # 10am
        "day_of_week":   1,      # Tuesday
        "is_weekend":    0,
    }

    # Empty history — first minute of data
    history = []

    decision, conf, proba_dict, metrics = predict(metrics_now, history)
    print_result(decision, conf, proba_dict, metrics)


# ══════════════════════════════════════════════════════════════
#  SCENARIO 2 — Three classic situations
# ══════════════════════════════════════════════════════════════
def scenario_three_cases():
    """
    Test the 3 expected behaviors:
    quiet night → scale_down
    normal day  → keep
    heavy load  → scale_up
    """
    print("\n" + "═" * 60)
    print("  SCENARIO 2 — Three Classic Situations")
    print("═" * 60)

    cases = {
        "Quiet night (2am)": {
            "cpu_mean": 0.05, "cpu_max": 0.10, "mem_mean": 0.05,
            "mem_max": 0.08, "wep": 3, "priority_mean": 50,
            "assigned_mem": 0.001, "page_cache": 0.0002,
            "hour": 2, "day_of_week": 2, "is_weekend": 0,
        },
        "Normal working hours (10am)": {
            "cpu_mean": 1.50, "cpu_max": 4.20, "mem_mean": 0.80,
            "mem_max": 2.50, "wep": 80, "priority_mean": 150,
            "assigned_mem": 0.010, "page_cache": 0.005,
            "hour": 10, "day_of_week": 0, "is_weekend": 0,
        },
        "Heavy load — CPU spike (2pm)": {
            "cpu_mean": 7.80, "cpu_max": 18.50, "mem_mean": 4.20,
            "mem_max": 12.00, "wep": 280, "priority_mean": 300,
            "assigned_mem": 0.020, "page_cache": 0.010,
            "hour": 14, "day_of_week": 1, "is_weekend": 0,
        },
    }

    # Use a small pre-filled history for each case
    for label, metrics_now in cases.items():
        history = [metrics_now.copy() for _ in range(3)]  # 3 min history
        decision, conf, proba_dict, metrics = predict(metrics_now, history)
        print(f"  🔹 {label}")
        print_result(decision, conf, proba_dict, metrics)


# ══════════════════════════════════════════════════════════════
#  SCENARIO 3 — Simulated traffic spike over 10 minutes
# ══════════════════════════════════════════════════════════════
def scenario_traffic_spike():
    """
    Simulate 10 minutes of traffic:
    - Minutes 1-3:  quiet baseline
    - Minutes 4-7:  traffic spike (CPU surges)
    - Minutes 8-10: traffic drops back to quiet

    Watch how the model reacts over time with temporal features.
    """
    print("\n" + "═" * 60)
    print("  SCENARIO 3 — Traffic Spike Over 10 Minutes")
    print("═" * 60)
    print("  Quiet (1-3) → Spike (4-7) → Recovery (8-10)\n")

    # Define CPU and workload per minute
    timeline = [
        # min  cpu    cpu_max   mem    wep   description
        (1,    0.08,  0.20,    0.06,  5,   "quiet baseline"),
        (2,    0.10,  0.25,    0.07,  6,   "quiet baseline"),
        (3,    0.09,  0.18,    0.06,  5,   "quiet baseline"),
        (4,    1.20,  3.50,    0.40,  40,  "traffic starting"),
        (5,    3.80,  9.10,    1.20,  120, "spike growing"),
        (6,    6.50,  15.20,   2.80,  210, "spike peak"),
        (7,    7.10,  17.80,   3.10,  240, "spike sustained"),
        (8,    4.20,  10.50,   2.10,  140, "traffic dropping"),
        (9,    1.10,  2.80,    0.60,  35,  "recovering"),
        (10,   0.12,  0.30,    0.08,  7,   "back to quiet"),
    ]

    history = []

    for (minute, cpu, cpu_max, mem, wep, desc) in timeline:
        metrics_now = {
            "cpu_mean":      cpu,
            "cpu_max":       cpu_max,
            "mem_mean":      mem,
            "mem_max":       mem * 1.5,
            "wep":           wep,
            "priority_mean": 150.0,
            "assigned_mem":  0.01,
            "page_cache":    0.005,
            "hour":          10 + minute,
            "day_of_week":   0,
            "is_weekend":    0,
        }

        decision, conf, proba_dict, metrics = predict(metrics_now, history)

        print(f"  ─── Minute {minute:2d} [{desc}] ───")
        print_result(decision, conf, proba_dict, metrics, minute=minute)

        # Update history buffer (keep last 6)
        history.append(metrics_now)
        if len(history) > 6:
            history.pop(0)

        time.sleep(0.3)  # small pause for readability


# ══════════════════════════════════════════════════════════════
#  SCENARIO 4 — Continuous loop simulation (real-time feel)
# ══════════════════════════════════════════════════════════════
def scenario_continuous():
    """
    Simulates the controller loop running every N seconds.
    CPU oscillates to show scale_up / keep / scale_down over time.
    Press Ctrl+C to stop.
    """
    print("\n" + "═" * 60)
    print("  SCENARIO 4 — Continuous Loop Simulation")
    print("  (simulates the 60-second controller loop)")
    print("  Press Ctrl+C to stop")
    print("═" * 60)

    INTERVAL = 3   # seconds between predictions (use 60 for real behavior)

    # CPU pattern: rises then falls then rises...
    cpu_pattern = [
        0.08, 0.10, 0.15, 0.80, 2.50, 5.10, 7.80, 8.20,
        7.50, 6.10, 4.20, 2.10, 0.90, 0.30, 0.10, 0.08,
    ]

    history = []
    minute  = 0

    try:
        while True:
            # Cycle through CPU pattern
            cpu     = cpu_pattern[minute % len(cpu_pattern)]
            cpu_max = cpu * 2.3 + np.random.uniform(0, 1.5)
            mem     = cpu * 0.4 + np.random.uniform(0, 0.2)
            wep     = int(cpu * 30 + np.random.uniform(0, 10))

            metrics_now = {
                "cpu_mean":      round(cpu, 3),
                "cpu_max":       round(cpu_max, 3),
                "mem_mean":      round(mem, 3),
                "mem_max":       round(mem * 1.4, 3),
                "wep":           wep,
                "priority_mean": 150.0,
                "assigned_mem":  0.01,
                "page_cache":    0.005,
                "hour":          (10 + minute) % 24,
                "day_of_week":   0,
                "is_weekend":    0,
            }

            decision, conf, proba_dict, metrics = predict(metrics_now, history)
            print_result(decision, conf, proba_dict, metrics, minute=minute+1)

            history.append(metrics_now)
            if len(history) > 6:
                history.pop(0)

            minute += 1
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n  Simulation stopped.")


# ══════════════════════════════════════════════════════════════
#  SCENARIO 5 — Custom metrics (you type the values)
# ══════════════════════════════════════════════════════════════
def scenario_custom():
    """
    You enter the CPU and workload values manually.
    The model predicts the decision.
    """
    print("\n" + "═" * 60)
    print("  SCENARIO 5 — Custom Metrics (you enter the values)")
    print("═" * 60)

    def get_float(prompt, default):
        try:
            val = input(f"  {prompt} [{default}]: ").strip()
            return float(val) if val else default
        except ValueError:
            return default

    def get_int(prompt, default):
        try:
            val = input(f"  {prompt} [{default}]: ").strip()
            return int(val) if val else default
        except ValueError:
            return default

    print("  Enter metric values (press Enter to use default)\n")
    cpu     = get_float("cpu_mean    (avg CPU %, e.g. 3.5)", 1.0)
    cpu_max = get_float("cpu_max     (peak CPU %, e.g. 8.2)", cpu * 2.5)
    mem     = get_float("mem_mean    (avg memory %, e.g. 1.2)", 0.5)
    wep     = get_int  ("wep         (events/min, e.g. 80)", 10)
    hour    = get_int  ("hour        (0-23, e.g. 10)", 10)

    metrics_now = {
        "cpu_mean":      cpu,
        "cpu_max":       cpu_max,
        "mem_mean":      mem,
        "mem_max":       mem * 1.5,
        "wep":           wep,
        "priority_mean": 150.0,
        "assigned_mem":  0.01,
        "page_cache":    0.005,
        "hour":          hour,
        "day_of_week":   0,
        "is_weekend":    int(hour < 6 or hour > 20),
    }

    # Simulate 3 minutes of history at the same level
    history = [metrics_now.copy() for _ in range(3)]

    decision, conf, proba_dict, metrics = predict(metrics_now, history)
    print()
    print_result(decision, conf, proba_dict, metrics)


# ══════════════════════════════════════════════════════════════
#  MAIN MENU
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 60)
    print("  AI KUBERNETES AUTOSCALING — LOCAL TEST")
    print("  Model: RandomForest  |  Classes: scale_up / keep / scale_down")
    print("═" * 60)

    scenarios = {
        "1": ("Single instant prediction",        scenario_single),
        "2": ("Three classic situations",         scenario_three_cases),
        "3": ("Traffic spike over 10 minutes",    scenario_traffic_spike),
        "4": ("Continuous loop simulation",       scenario_continuous),
        "5": ("Custom metrics (enter manually)",  scenario_custom),
        "0": ("Run ALL scenarios",                None),
    }

    print("\n  Choose a scenario:")
    for key, (label, _) in scenarios.items():
        print(f"    [{key}] {label}")

    choice = input("\n  Your choice: ").strip()

    if choice == "0":
        scenario_single()
        scenario_three_cases()
        scenario_traffic_spike()
    elif choice in scenarios and scenarios[choice][1]:
        scenarios[choice][1]()
    else:
        print("  Invalid choice. Running scenario 2 by default.")
        scenario_three_cases()

    print("═" * 60)
    print("  ✅ Test complete.")
    print("  Next step: make build → make load → make deploy → make logs")
    print("═" * 60)


if __name__ == "__main__":
    main()