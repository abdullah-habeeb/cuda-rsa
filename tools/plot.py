#!/usr/bin/env python3
"""
plot.py - Turns the benchmark CSV files in results/ into report figures (PNG).
Requires matplotlib.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# One colour per series, used consistently in every figure
C = {
    "gpu_crt": "#1f6f3f", "gpu_std": "#6fbf73",
    "cpuN_crt": "#1f4e9c", "cpuN_std": "#7aa6e0",
    "cpu1_crt": "#b3541e", "cpu1_std": "#f0a868",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
})


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def col(rows, key, typ=float):
    return [typ(r[key]) for r in rows]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote results/figures/" + name)


e1 = read("e1_scaling.csv")
N = col(e1, "connections", int)
T = int(e1[0]["cpu_threads"])
series = [
    ("gpu_crt", "gpu_crt_kernel_ms", "GPU  CRT"),
    ("gpu_std", "gpu_std_kernel_ms", "GPU  standard"),
    ("cpuN_crt", "cpuN_crt_ms", f"CPU {T} threads  CRT"),
    ("cpuN_std", "cpuN_std_ms", f"CPU {T} threads  standard"),
    ("cpu1_crt", "cpu1_crt_ms", "CPU 1 thread  CRT"),
    ("cpu1_std", "cpu1_std_ms", "CPU 1 thread  standard"),
]

# ---- Fig 1: time vs connections -------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for key, colname, label in series:
    ax.plot(N, col(e1, colname), marker="o", ms=3.5, lw=1.8, color=C[key], label=label,
            ls="-" if "crt" in key else "--")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Simultaneous connections (RSA-2048 decryptions)")
ax.set_ylabel("Time (ms, log scale)")
ax.set_title("Fig. 1  Decryption time vs. batch size")
ax.legend(fontsize=8, ncol=2)
save(fig, "fig1_time_vs_connections.png")

# ---- Fig 2: throughput -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for key, colname, label in series:
    thr = [n / (t / 1000.0) for n, t in zip(N, col(e1, colname))]
    ax.plot(N, thr, marker="o", ms=3.5, lw=1.8, color=C[key], label=label, ls="-" if "crt" in key else "--")
ax.set_xscale("log", base=2)
ax.set_xlabel("Simultaneous connections")
ax.set_ylabel("Throughput (decryptions / second)")
ax.set_title("Fig. 2  Throughput: the GPU needs many connections to fill up")
ax.legend(fontsize=8, ncol=2)
save(fig, "fig2_throughput.png")

# ---- Fig 3: speed-up of GPU (CRT) over CPU ---------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.8))
g = col(e1, "gpu_crt_kernel_ms")
ge = col(e1, "gpu_crt_e2e_ms")
c1 = col(e1, "cpu1_crt_ms")
cN = col(e1, "cpuN_crt_ms")
ax.plot(N, [a / b for a, b in zip(c1, g)], marker="o", ms=3.5, lw=1.8, color=C["cpu1_crt"],
        label="vs CPU 1 thread (kernel time)")
ax.plot(N, [a / b for a, b in zip(c1, ge)], marker="s", ms=3, lw=1.2, color=C["cpu1_crt"], ls=":",
        label="vs CPU 1 thread (incl. PCIe copies)")
ax.plot(N, [a / b for a, b in zip(cN, g)], marker="o", ms=3.5, lw=1.8, color=C["cpuN_crt"],
        label=f"vs CPU {T} threads (kernel time)")
ax.axhline(1.0, color="#555", lw=0.9)
ax.text(N[0], 1.04, "break-even", fontsize=7.5, color="#555")
ax.set_xscale("log", base=2)
ax.set_xlabel("Simultaneous connections")
ax.set_ylabel("Speed-up (x)")
ax.set_title("Fig. 3  GPU speed-up for CRT decryption")
ax.legend(fontsize=8)
save(fig, "fig3_speedup.png")

# ---- Fig 4: pipeline breakdown ----------------------------------------------
e2 = read("e2_breakdown.csv")
fig, ax = plt.subplots(figsize=(6.4, 2.6))
steps = [("alloc_ms", "1 cudaMalloc", "#999999"), ("h2d_ms", "2 Memcpy H->D", "#e0a030"),
         ("kernel_ms", "3 Kernel", "#1f6f3f"), ("d2h_ms", "4 Memcpy D->H", "#c05050"),
         ("free_ms", "5 cudaFree", "#666666")]
labels = []
for i, r in enumerate(e2):
    left = 0.0
    total = sum(float(r[k]) for k, _, _ in steps)
    for k, lab, colr in steps:
        v = float(r[k])
        ax.barh(i, v / total * 100, left=left, color=colr, label=lab if i == 0 else None, height=0.55)
        left += v / total * 100
    labels.append(f"{'CRT' if r['variant'] == 'crt' else 'Standard'}\n({r['connections']} conns)")
    ax.text(101, i, f"kernel = {float(r['kernel_ms']) / total * 100:.2f}% of {total:.1f} ms",
            va="center", fontsize=8)
ax.set_yticks(range(len(e2)))
ax.set_yticklabels(labels)
ax.set_xlim(0, 135)
ax.set_xlabel("Share of total GPU pipeline time (%)")
ax.set_title("Fig. 4  Where the time goes in the 5-step CUDA pipeline")
ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.42, -0.35))
ax.grid(axis="y", visible=False)
save(fig, "fig4_breakdown.png")

# ---- Fig 5: memory hierarchy ------------------------------------------------
e3 = read("e3_memory.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
names = [r["key_memory"] for r in e3]
vals = col(e3, "kernel_ms")
bars = ax.bar(names, vals, color=["#c05050", "#e0a030", "#1f6f3f"], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} ms", ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("Kernel time (ms)")
ax.set_xlabel("Memory space holding the RSA key (n, R^2, d)")
ax.set_title(f"Fig. 5  Key placement ({e3[0]['connections']} decryptions)")
ax.set_ylim(0, max(vals) * 1.15)
save(fig, "fig5_memory.png")

# ---- Fig 6: block size ------------------------------------------------------
e4 = read("e4_blocksize.csv")
fig, ax = plt.subplots(figsize=(5.8, 3.2))
blocks = [r["block"] for r in e4]
x = range(len(blocks))
s = col(e4, "std_kernel_ms")
c = col(e4, "crt_kernel_ms")
w = 0.38
ax.bar([i - w / 2 for i in x], [v if v > 0 else 0 for v in s], w, color=C["gpu_std"], label="standard")
ax.bar([i + w / 2 for i in x], [v if v > 0 else 0 for v in c], w, color=C["gpu_crt"], label="CRT")
for i, (vs, vc) in enumerate(zip(s, c)):
    if vs < 0:
        ax.text(i - w / 2, 5, "n/a", ha="center", fontsize=7, rotation=90)
    if vc < 0:
        ax.text(i + w / 2, 5, "n/a", ha="center", fontsize=7, rotation=90)
ax.set_xticks(list(x))
ax.set_xticklabels(blocks)
ax.set_xlabel("Threads per block")
ax.set_ylabel("Kernel time (ms)")
ax.set_title(f"Fig. 6  Effect of block size ({e4[0]['connections']} decryptions)")
ax.legend(fontsize=8)
save(fig, "fig6_blocksize.png")
print("all figures written")
