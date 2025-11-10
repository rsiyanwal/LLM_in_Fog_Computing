## Copying and running a bash file on all the Pis
```bash
while IFS= read -r h; do [ -z "$h" ] && continue; echo "== Copying and running on $h =="; scp -q systemprep_pi.sh pi@"$h":~ || { echo "scp failed for $h"; continue; }; ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new pi@"$h" 'bash ~/systemprep_pi.sh' || echo "ssh/script failed on $h"; done < hosts.txt
```
one-time check:
`~/edge-llm-bench/bin/run_one.sh blas ~/edge-llm-bench/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ~/edge-llm-bench/prompts/sum.txt 4 512 32 greedy`

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

### Attaching energy sensor to Pi
Sensor used: WCMCU-3221
Install:
```
sudo apt update
sudo apt install -y raspi-config
sudo apt install -y i2c-tools python3-pip
pip install adafruit-circuitpython-ina3221
pip install adafruit-blinka
```
Then run:
```
sudo raspi-config
```
Interface Options → I2C → Enable
Edit the file:
```
sudo nano /boot/firmware/config.txt
dtparam=i2c_arm=on
```
Attach the sensor as follows:
1. Connect the SDA pin, SCL pin, VCC (VS) pin, and GND pin of the sensor to the pins SDA pin (GPIO2), SCL pin (GPIO3), 3.3V or 5V pin, and GND pin of the Raspberry Pi, respectively.
2. Detect the sensor: `sudo i2cdetect -y 1`
You may see an output like this:
```
(venv) pi@pi01:~ $ sudo i2cdetect -y 1
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
40: 40 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- --
```
This command scans all possible I2C addresses. The output means that the sensor is detected at the address 0x40. 
Use the minimal code to test:
```python
import board
import busio
from adafruit_ina3221 import INA3221

i2c = busio.I2C(board.SCL, board.SDA)
ina = INA3221(i2c)

for i, ch in enumerate(ina):
    print(f"--- Channel {i+1} ---")
    print(f"Bus Voltage:   {ch.bus_voltage:.2f} V")
    print(f"Shunt Voltage: {ch.shunt_voltage*1000:.4f} mV")
    print(f"Current:       {ch.current:.2f} mA")
    print("---------------------")

```
Sometimes, you may need to install on the base image:
```
sudo pip3 install --break-system-packages adafruit-blinka adafruit-circuitpython-ina3221
```
