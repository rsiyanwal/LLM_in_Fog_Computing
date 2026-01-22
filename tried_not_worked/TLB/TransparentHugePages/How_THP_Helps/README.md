# How THP Helps
**MAKE SURE IT IS ENABLED!**

We will test before and after THP is enabled.
**STEP 1: Create a directory**
```
mkdir -p ~/thp-experiment
cd ~/thp-experiment
```
**STEP 2: Define a workload once**
Install stress tool: `sudo apt install -y stress-ng`
Define workload: `WORKLOAD="stress-ng --vm 1 --vm-bytes 4G --vm-method all --timeout 60s"`

**STEP 3: Before THP (baseline)**
Disable THP:
```
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```
Clear caches:
```
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```
Save runtime output:
```
time $WORKLOAD > before_runtime.txt 2>&1
```
Save TLB statistics:
```
perf stat \
  -e dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses \
  $WORKLOAD > before_perf.txt 2>&1
```
In another terminal, simultaneously save the memory state: `grep AnonHugePages /proc/meminfo > before_meminfo.txt

**STEP 4: After THP**
Enable THP:
```
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```
Clear cache again:
```
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```
Save runtime: `time $WORKLOAD > after_runtime.txt 2>&1`
Save stats:
```
perf stat \
  -e dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses \
  $WORKLOAD > after_perf.txt 2>&1
```
In another terminal, simultaneously save the memory state: `grep AnonHugePages /proc/meminfo > after_meminfo.txt`

**STEP 5: Compare before and after**
Compare runtime: `diff before_runtime.txt after_runtime.txt`
Compare TLB misses: `diff before_perf.txt after_perf.txt` (focus on dTLB-load-misses)
Compare huge page usage:
```
cat before_meminfo.txt
cat after_meminfo.txt
```

OUTPUT:
```python
pi05@raspberrypi:~/thp-experiment $ diff before_runtime.txt after_runtime.txt
1,8c1,8
< stress-ng: info:  [13389] setting to a 1 min run per stressor
< stress-ng: info:  [13389] dispatching hogs: 1 vm
< stress-ng: info:  [13396] vm: using 4G per stressor instance (total 4G of 7.13G available memory)
< stress-ng: info:  [13389] skipped: 0
< stress-ng: info:  [13389] passed: 1: vm (1)
< stress-ng: info:  [13389] failed: 0
< stress-ng: info:  [13389] metrics untrustworthy: 0
< stress-ng: info:  [13389] successful run completed in 1 min
---
> stress-ng: info:  [14675] setting to a 1 min run per stressor
> stress-ng: info:  [14675] dispatching hogs: 1 vm
> stress-ng: info:  [14676] vm: using 4G per stressor instance (total 4G of 7.12G available memory)
> stress-ng: info:  [14675] skipped: 0
> stress-ng: info:  [14675] passed: 1: vm (1)
> stress-ng: info:  [14675] failed: 0
> stress-ng: info:  [14675] metrics untrustworthy: 0
> stress-ng: info:  [14675] successful run completed in 1 min
pi05@raspberrypi:~/thp-experiment $ diff before_perf.txt after_perf.txt
1,8c1,8
< stress-ng: info:  [13926] setting to a 1 min run per stressor
< stress-ng: info:  [13926] dispatching hogs: 1 vm
< stress-ng: info:  [13927] vm: using 4G per stressor instance (total 4G of 7.11G available memory)
< stress-ng: info:  [13926] skipped: 0
< stress-ng: info:  [13926] passed: 1: vm (1)
< stress-ng: info:  [13926] failed: 0
< stress-ng: info:  [13926] metrics untrustworthy: 0
< stress-ng: info:  [13926] successful run completed in 1 min
---
> stress-ng: info:  [15191] setting to a 1 min run per stressor
> stress-ng: info:  [15191] dispatching hogs: 1 vm
> stress-ng: info:  [15192] vm: using 4G per stressor instance (total 4G of 7.11G available memory)
> stress-ng: info:  [15191] skipped: 0
> stress-ng: info:  [15191] passed: 1: vm (1)
> stress-ng: info:  [15191] failed: 0
> stress-ng: info:  [15191] metrics untrustworthy: 0
> stress-ng: info:  [15191] successful run completed in 1 min
13c13
<         19,598,096      dTLB-load-misses:u
---
>            550,141      dTLB-load-misses:u
15c15
<              4,907      iTLB-load-misses:u
---
>              5,035      iTLB-load-misses:u
17c17
<       60.637565972 seconds time elapsed
---
>       60.083144776 seconds time elapsed
19,20c19,20
<       32.494286000 seconds user
<       27.847053000 seconds sys
---
>       53.304885000 seconds user
>        6.483855000 seconds sys
pi05@raspberrypi:~/thp-experiment $ cat before_meminfo.txt
AnonHugePages:     24576 kB
pi05@raspberrypi:~/thp-experiment $ cat after_meminfo.txt
AnonHugePages:   4182016 kB
```
OBSERVATION:
1. ~19.6 million to 0.55 million TLB misses (~35x reduction)
2. User time (time spent by the CPU in executing the user's program instructions), Kernel time (time spent executing OS code on your behalf). In this case, kernel time drops by ~21 seconds. THP fundamentally changed where time is spent.
3. Before, almost all memory was 4KB pages, but after the entire workload was backed by 2MB huge pages.




