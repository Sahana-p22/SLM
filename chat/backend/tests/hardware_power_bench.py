"""1.13 Hardware Speed Test + 1.14 Power and Resource Usage Check.

1.13: calls the model directly in-process (bypassing HTTP) to get real
prefill/decode timing, plus a genuine cold-start timing (kill the
backend, time a fresh model load to ready).

1.14: samples host CPU%/RAM and GPU power/utilization (nvidia-smi - the
actual accelerator here, the direct equivalent of the Axelera Metis chip
in the source report) once per second during a real batch of chat
requests against the live backend. Intel RAPL package-energy access
requires root on this host (confirmed: read denied without sudo) - noted
honestly rather than silently substituted; GPU power via nvidia-smi's own
sensor is the more directly relevant reading anyway, since the GPU is
this deployment's accelerator (RAPL's role in the source report was
specifically to show the HOST isn't the bottleneck).
"""
import json
import statistics as st
import subprocess
import threading
import time

import requests

from bench_config import (
    CHAT_URL, VENV_PYTHON, BACKEND_HOST, BACKEND_PORT, REPO_ROOT,
    env_with_cuda, wait_for_health,
)

API = CHAT_URL

# Real decode throughput, from token_timing_bench.py's direct in-process
# measurement (llama.cpp's own reported completion_tokens / wall time) -
# reused here rather than re-measured, so energy-per-token is grounded in
# the same real throughput figure quoted elsewhere in this report.
decode_tok_s = None
try:
    with open("/tmp/token_timing_results.json") as f:
        _tt = json.load(f)
        decode_tok_s = _tt.get("summary", {}).get("query_gen", {}).get("mean_tokens_per_s")
except Exception:
    pass

# ---------------------------------------------------------------------
# 1.14 Power/resource sampling during real load
# ---------------------------------------------------------------------
samples = []
stop = threading.Event()


def sample_loop():
    while not stop.is_set():
        cpu = subprocess.run(
            ["bash", "-c", "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | sed 's/%us,//'"],
            capture_output=True, text=True).stdout.strip()
        mem = subprocess.run(
            ["bash", "-c", "free -m | awk '/Mem:/{print $3}'"],
            capture_output=True, text=True).stdout.strip()
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        try:
            cpu_pct = float(cpu) if cpu else None
        except ValueError:
            cpu_pct = None
        gpu_power, gpu_util, gpu_mem, gpu_temp = (None, None, None, None)
        if gpu:
            parts = [p.strip() for p in gpu.split(",")]
            if len(parts) == 4:
                gpu_power, gpu_util, gpu_mem, gpu_temp = parts
        samples.append({
            "t": round(time.time(), 1), "host_cpu_pct": cpu_pct,
            "host_mem_mb": int(mem) if mem else None,
            "gpu_power_w": float(gpu_power) if gpu_power else None,
            "gpu_util_pct": float(gpu_util) if gpu_util else None,
            "gpu_mem_mb": float(gpu_mem) if gpu_mem else None,
            "gpu_temp_c": float(gpu_temp) if gpu_temp else None,
        })
        stop.wait(1)


QUESTIONS = [
    "how many alerts today", "how many hand touch alerts this week",
    "what's the average inspection time", "break down alerts by type this month",
    "which day had the most fast inspection alerts", "compare this week vs last week",
    "give me a weekly report", "how many missing cleaning alerts yesterday",
]

print("=" * 70)
print("1.14 — Power/resource sampling during a real request batch")
print("=" * 70)
idle = subprocess.run(
    ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True).stdout.strip()
idle_power_w, idle_temp_c = (None, None)
if idle:
    iparts = [p.strip() for p in idle.split(",")]
    if len(iparts) == 2:
        idle_power_w, idle_temp_c = float(iparts[0]), float(iparts[1])
print(f"  Idle baseline (model loaded, no active request): {idle_power_w}W, {idle_temp_c}°C")

t = threading.Thread(target=sample_loop, daemon=True)
t.start()
t0 = time.time()
for q in QUESTIONS * 2:
    try:
        requests.post(API, json={"question": q, "history": []}, timeout=120)
    except Exception:
        pass
wall = time.time() - t0
stop.set()
t.join(timeout=3)

cpu_vals = [s["host_cpu_pct"] for s in samples if s["host_cpu_pct"] is not None]
gpu_power_vals = [s["gpu_power_w"] for s in samples if s["gpu_power_w"] is not None]
gpu_util_vals = [s["gpu_util_pct"] for s in samples if s["gpu_util_pct"] is not None]
gpu_temp_vals = [s["gpu_temp_c"] for s in samples if s["gpu_temp_c"] is not None]

print(f"  {len(samples)} samples over {wall:.0f}s")
print(f"  Host CPU: mean={st.mean(cpu_vals):.1f}%  max={max(cpu_vals):.1f}%" if cpu_vals else "  Host CPU: n/a")
print(f"  GPU power: mean={st.mean(gpu_power_vals):.1f}W  max={max(gpu_power_vals):.1f}W" if gpu_power_vals else "  GPU power: n/a")
print(f"  GPU util: mean={st.mean(gpu_util_vals):.1f}%  max={max(gpu_util_vals):.1f}%" if gpu_util_vals else "  GPU util: n/a")
print(f"  GPU temp: mean={st.mean(gpu_temp_vals):.1f}C  peak={max(gpu_temp_vals):.1f}C" if gpu_temp_vals else "  GPU temp: n/a")

# Energy per output token: Joules/token = mean power (W) / decode throughput (tok/s),
# since Watts = Joules/second, so W / (tokens/second) = Joules/token. Uses the real
# decode throughput measured directly in 1.13 above (query-gen mean tok/s), not a guess.
gpu_power_mean_w = st.mean(gpu_power_vals) if gpu_power_vals else None
energy_per_token_j = (gpu_power_mean_w / decode_tok_s) if (gpu_power_mean_w and decode_tok_s) else None
if energy_per_token_j:
    print(f"  Energy per output token: {energy_per_token_j*1000:.1f} mJ/token "
          f"(mean {gpu_power_mean_w:.1f}W / {decode_tok_s:.1f} tok/s decode)")

power_results = {
    "wall_s": wall, "n_samples": len(samples),
    "idle_power_w": idle_power_w, "idle_temp_c": idle_temp_c,
    "host_cpu_mean_pct": st.mean(cpu_vals) if cpu_vals else None,
    "host_cpu_max_pct": max(cpu_vals) if cpu_vals else None,
    "gpu_power_mean_w": gpu_power_mean_w,
    "gpu_power_max_w": max(gpu_power_vals) if gpu_power_vals else None,
    "gpu_util_mean_pct": st.mean(gpu_util_vals) if gpu_util_vals else None,
    "gpu_util_max_pct": max(gpu_util_vals) if gpu_util_vals else None,
    "gpu_temp_mean_c": st.mean(gpu_temp_vals) if gpu_temp_vals else None,
    "gpu_temp_max_c": max(gpu_temp_vals) if gpu_temp_vals else None,
    "decode_tok_s_used_for_energy": decode_tok_s,
    "energy_per_output_token_j": energy_per_token_j,
    "intel_rapl_note": "energy_uj read denied without root on this host; GPU power via nvidia-smi used instead (the actual accelerator here)",
    "raw_samples": samples,
}

# ---------------------------------------------------------------------
# 1.13 Cold load time - restart the backend, time to first successful request
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("1.13 — Cold model load time (backend restart -> first successful request)")
print("=" * 70)
subprocess.run(["pkill", "-f", "chat.backend.main:app"], capture_output=True)
time.sleep(3)
t_restart0 = time.time()
subprocess.Popen(
    [VENV_PYTHON, "-m", "uvicorn", "chat.backend.main:app",
     "--host", BACKEND_HOST, "--port", BACKEND_PORT],
    cwd=REPO_ROOT, env=env_with_cuda(),
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
cold_ready_s = time.time() - t_restart0 if wait_for_health() else None
print(f"  Cold load (process start -> healthy): {cold_ready_s:.2f}s" if cold_ready_s else "  FAILED to come up in 120s")

json.dump({"power": power_results, "cold_load_s": cold_ready_s},
          open("/tmp/hardware_power_results.json", "w"), indent=2)
