# Trying Transparent HugePages on Raspberry Pi 4B 8GB
OS: Raspberry Pi 64-bit OS (Bookworm)
```
CONFIG_HUGETLBFS=y
CONFIG_HUGETLB_PAGE=y
CONFIG_ARCH_WANT_HUGE_PMD_SHARE=y
CONFIG_TRANSPARENT_HUGEPAGE=y
```
```
sudo apt update
sudo apt install -y git bc bison flex libssl-dev make
sudo apt install -y libncurses-dev pkg-config
git clone https://github.com/raspberrypi/linux
cd linux
KERNEL=kernel8
make bcm2711_defconfig
make menuconfig
```
Search for `Transparent Hugepage Support` and enable it.
Check for the `Image`, you shouldn't be able to see it before building the kernel:
```
ls -lh arch/arm64/boot/
```
Build the kernel:
```
make -j$(nproc) Image modules dtbs
```
Check for the image again:
```
pi@pi03:~/linux $ ls -lh arch/arm64/boot/
total 27M
drwxr-xr-x 37 pi pi 4.0K Jan 22 17:11 dts
-rw-r--r--  1 pi pi  27M Jan 22 19:07 Image
-rwxr-xr-x  1 pi pi 1009 Jan 21 20:47 install.sh
-rw-r--r--  1 pi pi 1.6K Jan 21 20:47 Makefile
```
Check uname before:
```
pi05@raspberrypi:~/linux $ uname -a
Linux raspberrypi 6.12.62+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.12.62-1+rpt1 (2025-12-18) aarch64 GNU/Linux
```
Manually install it to the firmware boot path:
```
sudo cp arch/arm64/boot/Image /boot/firmware/kernel8.img
sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/firmware/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/firmware/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/firmware/overlays/
```
Install: 
```
sudo make modules_install
sudo make install
```
Check uname after reboot:
```
pi@pi03:~ $ uname -a
Linux pi03 6.12.66-v8+ #1 SMP PREEMPT Thu Jan 22 19:07:15 IST 2026 aarch64 GNU/Linux
```
Check the rest of the things:
```
pi@pi03:~ $ cd linux/
pi@pi03:~/linux $ ls /sys/kernel/mm/
lru_gen  mempolicy  numa  swap  transparent_hugepage
pi@pi03:~/linux $ ls /sys/kernel/mm/transparent_hugepage/
defrag   hpage_pmd_size    hugepages-128kB  hugepages-2048kB  hugepages-32kB   hugepages-64kB  khugepaged     shrink_underused
enabled  hugepages-1024kB  hugepages-16kB   hugepages-256kB   hugepages-512kB  hugepages-8kB   shmem_enabled  use_zero_page
pi@pi03:~/linux $ grep AnonHugePages /proc/meminfo
AnonHugePages:      6144 kB
```
Next, check: [How_THP_Helps](https://github.com/rsiyanwal/LLM_in_Fog_Computing/blob/main/tried_not_worked/TLB/TransparentHugePages/How_THP_Helps/README.md)


