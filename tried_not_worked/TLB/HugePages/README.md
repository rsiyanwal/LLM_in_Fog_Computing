# Trying HugePages on Raspberry Pi 4B 8GB
**OS:** Raspberry Pi 64-bit OS (Bookworm)
**File:** `/sys/kernel/mm/hugepages/*` not available. Signifying either that hugepages are disabled or not available on the hardware. 
```
pi05@raspberrypi:~ $ sudo find /sys -type d -iname '*huge*' 2>/dev/null
pi05@raspberrypi:~ $ uname -a
Linux raspberrypi 6.12.62+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.12.62-1+rpt1 (2025-12-18) aarch64 GNU/Linux
pi05@raspberrypi:~ $ grep huge /proc/m
meminfo  misc     modules  mounts
pi05@raspberrypi:~ $ grep huge /proc/meminfo
pi05@raspberrypi:~ $ grep Huge /proc/meminfo
pi05@raspberrypi:~ $ sudo sysctl -w vm.nr_hugepages=128
sysctl: cannot stat /proc/sys/vm/nr_hugepages: No such file or directory
pi05@raspberrypi:~ $
```
Same result for all the Pi: pi03, pi04, pi05, and pi06.

## Trying rebuilding kernel with HugePages
Required kernel options:
```
CONFIG_HUGETLBFS=y
CONFIG_HUGETLB_PAGE=y
CONFIG_ARCH_WANT_HUGE_PMD_SHARE=y
CONFIG_TRANSPARENT_HUGEPAGE=y
```
```
sudo apt update
sudo apt install git bc bison flex libssl-dev make
sudo apt install -y libncurses-dev pkg-config
git clone https://github.com/raspberrypi/linux
cd linux
KERNEL=kernel8
make bcm2711_defconfig
make menuconfig
```
<img width="1361" height="629" alt="menu_to_enable_hugepages" src="https://github.com/user-attachments/assets/c7daed32-cc4c-40f5-9a00-58121a1ffe0b" />
Search for: `HugeTLB filesystem support`, `HugeTLB page support`, `Transparent Hugepage Support`.
They are not available on the hardware. 
