import numpy as np

def safe_get(series, n):
    return series[-n] if len(series) >= n else series[0]

def compute_temporal(current, history):
    """
    Recompute the 13 temporal features from history buffer.
    MUST match exactly what was done in the Colab training notebook.
    """
    cpu = [h["cpu_mean"] for h in history] + [current["cpu_mean"]]
    wep = [h["wep"]      for h in history] + [current["wep"]]

    # Lag features
    current["cpu_lag1"] = safe_get(cpu, 2)
    current["cpu_lag3"] = safe_get(cpu, 4)
    current["cpu_lag5"] = safe_get(cpu, 6)
    current["wep_lag1"] = safe_get(wep, 2)
    current["wep_lag3"] = safe_get(wep, 4)

    # Rolling window (5 minutes)
    last5 = cpu[-5:]
    current["cpu_roll5_mean"] = float(np.mean(last5))
    current["cpu_roll5_max"]  = float(np.max(last5))
    current["cpu_roll5_std"]  = float(np.std(last5))
    current["wep_roll5_mean"] = float(np.mean(wep[-5:]))

    # Trend and acceleration
    current["cpu_trend"] = cpu[-1] - safe_get(cpu, 4)
    current["wep_trend"] = wep[-1] - safe_get(wep, 4)

    # Spike persistence
    current["spike_persist"] = int(
        current["cpu_max"] > current["cpu_roll5_mean"] * 2)

    # Acceleration
    prev_trend = (cpu[-2] - safe_get(cpu, 5)) if len(cpu) >= 5 else 0
    current["cpu_accel"] = current["cpu_trend"] - prev_trend

    return current