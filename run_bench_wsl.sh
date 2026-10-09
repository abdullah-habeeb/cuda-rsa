#!/usr/bin/env bash
# Builds and runs the full experiment detached from the terminal (WSL).
cd "$(dirname "$0")"
bash build_wsl.sh >/dev/null && ./rsa_cuda demo 8 > results/demo_log.txt 2>&1 \
  && ./rsa_cuda verify 1000 > results/verify_log.txt 2>&1 \
  && ./rsa_cuda bench > results/bench_log.txt 2>&1
echo "exit=$?" > results/DONE
