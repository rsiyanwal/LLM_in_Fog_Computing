## Copying and running a bash file on all the Pis
```bash
while IFS= read -r h; do [ -z "$h" ] && continue; echo "== Copying and running on $h =="; scp -q systemprep_pi.sh pi@"$h":~ || { echo "scp failed for $h"; continue; }; ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new pi@"$h" 'bash ~/systemprep_pi.sh' || echo "ssh/script failed on $h"; done < hosts.txt
```
one-time check:
~/edge-llm-bench/bin/run_one.sh blas   ~/edge-llm-bench/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf   ~/edge-llm-bench/prompts/sum.txt 4 512 32 greedy

During the runtime of the command above, you may encounter an error such as a name mismatch. You can sort it via:
```bash
cd ~/edge-llm-bench/bin
ln -s llama_blas main_blas
ln -s llama_pure main_pure
chmod +x llama_blas llama_pure main_blas main_pure run_one.sh
```


