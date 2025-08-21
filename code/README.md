## Copying and running a bash file on all the Pis
```bash
while IFS= read -r h; do [ -z "$h" ] && continue; echo "== Copying and running on $h =="; scp -q systemprep_pi.sh pi@"$h":~ || { echo "scp failed for $h"; continue; }; ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new pi@"$h" 'bash ~/systemprep_pi.sh' || echo "ssh/script failed on $h"; done < hosts.txt
```
one-time check:
~/edge-llm-bench/bin/run_one.sh blas ~/edge-llm-bench/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ~/edge-llm-bench/prompts/sum.txt 4 512 32 greedy

During the runtime of the command above, you may encounter an error such as a name mismatch. You can sort it via:
```bash
cd ~/edge-llm-bench/bin
ln -s llama_blas main_blas
ln -s llama_pure main_pure
chmod +x llama_blas llama_pure main_blas main_pure run_one.sh
```

You may get an illegal instruction error:
```bash
# sanity: interpreter itself?
python3 -c 'print("ok")'

# likely culprits (native extensions):
python3 -c 'import numpy as np; print("numpy", np.__version__)'
python3 -c 'import pandas as pd; print("pandas", pd.__version__)'
python3 -c 'import pyarrow as pa; print("pyarrow", pa.__version__)'

# pure-Python (should not SIGILL; if they do, it’s an indirect dependency)
python3 -c 'import sacrebleu; print("sacrebleu", sacrebleu.__version__)'
python3 -c 'from rouge_score import rouge_scorer; print("rouge_score ok")'
python3 -c 'from datasets import load_dataset; print("datasets ok")'
```
Check one by one, find the culprit. Then uninstall and install it again:
```bash
python3 -m pip uninstall -y pandas pyarrow datasets
python3 -m pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple   pandas==2.1.4 pyarrow==12.0.1 datasets==2.14.6
```
CPU governer for speed-up
```bash
sudo apt-get install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl disable --now ondemand || true
sudo systemctl enable --now cpufrequtils
```
