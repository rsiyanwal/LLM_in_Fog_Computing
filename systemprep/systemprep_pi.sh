#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y git build-essential cmake python3 python3-venv python3-pip \
  jq bc pkg-config libopenblas-dev htop lm-sensors curl wget unzip tmux \
  libraspberrypi-bin  # vcgencmd lives here on RPi OS

# Optional: keep clocks stable for consistent latency (not mandatory)
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$g" >/dev/null || true
done

mkdir -p ~/edge-llm-bench/{bin,models,prompts,logs,results,tmp}
