import subprocess, time, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
ram_pattern = re.compile(r"RAM (\d+)/(\d+)MB")
gpu_pattern = re.compile(r"GR3D_FREQ (\d+)%@(\d+)")
temp_pattern = re.compile(r"thermal@(\d+\.?\d*)C")

MAX_TEMP = 70  # °C

def run_experiment(bs):
    print(f"\n=== Batch size {bs} ===")
    log = []
    start = time.time()

    tegra = subprocess.Popen(
        ["sudo", "tegrastats", "--interval", "1000"],
        stdout=subprocess.PIPE, universal_newlines=True
    )
    train = subprocess.Popen(
        ["bash", "run.sh", "-t", ".", "-n", "LeNet", "-ln", "0",
         "-bs", str(bs), "-trt", "-sz", "7"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    try:
        for line in tegra.stdout:
            t = time.time() - start
            ram_match = ram_pattern.search(line)
            gpu_match = gpu_pattern.search(line)
            temp_match = temp_pattern.search(line)
            if ram_match and gpu_match and temp_match:
                ram_used = int(ram_match.group(1))
                gpu_util = int(gpu_match.group(1))
                temp = float(temp_match.group(1))
                log.append((t, ram_used, gpu_util, temp))
                print(f"[bs={bs}] {t:.1f}s → RAM={ram_used}MB, GPU={gpu_util}%, Temp={temp}°C")

                # Thermal control
                if temp > MAX_TEMP:
                    print(f"⚠ Thermal limit reached ({temp}°C). Stopping batch {bs}.")
                    train.terminate()
                    break

            if train.poll() is not None:
                break
    finally:
        tegra.kill()
        train.wait()

    return pd.DataFrame(log, columns=["time", "ram", "gpu", "temp"])

def analyze(df, bs):
    peak_ram = df["ram"].max()
    avg_ram = df["ram"].mean()
    peak_gpu = df["gpu"].max()
    avg_gpu = df["gpu"].mean()
    peak_temp = df["temp"].max()
    print(f"\n📊 Batch size {bs} summary:")
    print(f"- RAM: peak={peak_ram}MB, avg={avg_ram:.1f}MB")
    print(f"- GPU: peak={peak_gpu}%, avg={avg_gpu:.1f}%")
    print(f"- Temp: peak={peak_temp}°C")
    return df, avg_gpu

if __name__ == "__main__":
    results = {}
    gpu_avgs = {}
    for bs in batch_sizes:
        df = run_experiment(bs)
        df, avg_gpu = analyze(df, bs)
        results[bs] = df
        gpu_avgs[bs] = avg_gpu

    # Plot RAM
    plt.figure(figsize=(10, 5))
    for bs, df in results.items():
        plt.plot(df["time"], df["ram"], label=f"bs {bs}")
    plt.xlabel("Time (s)"); plt.ylabel("RAM (MB)")
    plt.title("RAM usage vs. Time"); plt.legend(); plt.grid()
    plt.savefig("ram_comparison.png")

    # Plot GPU
    plt.figure(figsize=(10, 5))
    for bs, df in results.items():
        plt.plot(df["time"], df["gpu"], label=f"bs {bs}")
    plt.xlabel("Time (s)"); plt.ylabel("GPU Utilization (%)")
    plt.title("GPU usage vs. Time"); plt.legend(); plt.grid()
    plt.savefig("gpu_comparison.png")

    # Suggest optimal batch size
    optimal_bs = max((bs for bs, avg in gpu_avgs.items() if results[bs]["ram"].max() < 3500),
                     key=lambda bs: gpu_avgs[bs])
    print(f"\n✅ Suggested optimal batch size: {optimal_bs}")
    print("Plots saved: ram_comparison.png, gpu_comparison.png")
