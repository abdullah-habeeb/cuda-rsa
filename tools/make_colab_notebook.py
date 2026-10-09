#!/usr/bin/env python3
"""Builds RSA_CUDA_Colab.ipynb: a self-contained Google Colab notebook with every source file embedded."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["src/bigint.cuh", "src/rsa_key.h", "src/rsa_cuda.cu", "Makefile", "tools/verify.py", "tools/plot.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    md("# CUDA Case Study - Parallel RSA-2048 Encryption / Decryption\n"
       "**Abdullah (1BY23CS002) - Parallel Computing (BCS702) CCA 1**\n\n"
       "How to run:\n"
       "1. **Runtime -> Change runtime type -> T4 GPU -> Save**\n"
       "2. **Runtime -> Run all**\n\n"
       "The notebook writes the project files, compiles them with `nvcc` for whatever GPU Colab gives you, "
       "runs the narrated demo (the 5 CUDA steps), the correctness suite, and an independent Python check. "
       "The full benchmark at the end is optional (about 10 minutes)."),
    code("# 1. Check the GPU and pick the matching CUDA architecture\n"
         "import os, subprocess\n"
         "os.environ['PATH'] += ':/usr/local/cuda/bin'\n"
         "r = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],\n"
         "                   capture_output=True, text=True)\n"
         "if r.returncode != 0:\n"
         "    raise SystemExit('No GPU found: use Runtime -> Change runtime type -> T4 GPU, then run again.')\n"
         "name, cc = [x.strip() for x in r.stdout.strip().splitlines()[0].split(',')]\n"
         "ARCH = 'sm_' + cc.replace('.', '')\n"
         "print(f'GPU: {name}  (compute capability {cc} -> {ARCH})')\n"
         "!nvcc --version | tail -2"),
    code("# 2. Project folder\n"
         "os.makedirs('/content/CUDA_RSA_CCA/src', exist_ok=True)\n"
         "os.makedirs('/content/CUDA_RSA_CCA/tools', exist_ok=True)\n"
         "%cd /content/CUDA_RSA_CCA"),
    md("## Project source files\n"
       "`bigint.cuh` = 2048-bit Montgomery arithmetic (shared by CPU and GPU), `rsa_key.h` = demo RSA-2048 key, "
       "`rsa_cuda.cu` = kernels + 5-step host pipeline + demo/verify/bench modes."),
]
for f in FILES:
    with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
        body = fh.read()
    cells.append(code(f"%%writefile {f}\n" + body))
cells += [
    md("## Build\nThe first compile takes about a minute (the 2048-bit arithmetic is fully unrolled)."),
    code("!make ARCH={ARCH} 2>&1 | grep -v '^ulimit' | tail -5\n!ls -la rsa_cuda"),
    md("## Demo: 8 simultaneous HTTPS connections\n"
       "Clients encrypt session keys with the server's public key; the server decrypts all of them at once on "
       "the GPU, one thread per connection. Watch for the `[Step 1]` ... `[Step 5]` lines."),
    code("!./rsa_cuda info\n!./rsa_cuda demo 8"),
    md("## Correctness suite + independent check\n"
       "GPU vs CPU vs original message on 1,000 messages, the three key-memory variants, CRT, and odd batch sizes "
       "(boundary check). Then Python's own `pow()` re-checks every GPU result."),
    code("!./rsa_cuda verify 1000\n!python3 tools/verify.py"),
    md("## Optional: full benchmark (about 10 minutes on Colab)\n"
       "Uncomment and run. Colab's CPU has only 2 cores, so the CPU baseline is slower than on a laptop."),
    code("# !./rsa_cuda bench\n"
         "# !python3 tools/plot.py\n"
         "# from IPython.display import Image, display\n"
         "# for f in sorted(os.listdir('results/figures')): display(Image('results/figures/' + f))"),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(ROOT, "RSA_CUDA_Colab.ipynb")
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print("wrote", out, f"({os.path.getsize(out) // 1024} KB, {len(cells)} cells)")
