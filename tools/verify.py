#!/usr/bin/env python3
"""
verify.py - Independent correctness check of the CUDA results.

Reads results/verify_dump.txt written by `rsa_cuda verify` (the key and every
(plaintext m, GPU ciphertext c) pair) and re-computes both directions with
Python's built-in arbitrary-precision pow(), which shares no code with the
CUDA/C++ implementation:
    pow(m, e, n) == c      (GPU encryption is correct)
    pow(c, d, n) == m      (the ciphertext really decrypts to the plaintext)
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "verify_dump.txt")

with open(path) as f:
    lines = [ln.split() for ln in f if ln.strip()]

hdr = {k: v for k, v in lines[:3]}
n, e, d = int(hdr["n"], 16), int(hdr["e"], 16), int(hdr["d"], 16)
pairs = [(int(a, 16), int(b, 16)) for a, b in lines[3:]]

t0 = time.perf_counter()
enc_bad = sum(pow(m, e, n) != c for m, c in pairs)
dec_bad = sum(pow(c, d, n) != m for m, c in pairs)
dt = time.perf_counter() - t0

print(f"Independent Python check of {len(pairs)} GPU results (RSA-{n.bit_length()}):")
print(f"  pow(m, e, n) == c_gpu : {len(pairs) - enc_bad}/{len(pairs)} {'PASS' if enc_bad == 0 else 'FAIL'}")
print(f"  pow(c, d, n) == m     : {len(pairs) - dec_bad}/{len(pairs)} {'PASS' if dec_bad == 0 else 'FAIL'}")
print(f"  (Python single-thread time for {len(pairs)} decryptions: {dt:.2f} s)")
sys.exit(0 if enc_bad == dec_bad == 0 else 1)
