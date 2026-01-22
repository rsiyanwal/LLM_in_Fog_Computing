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
----> STOP!
sudo make modules_install
sudo make install
```
Check uname before:
```
pi05@raspberrypi:~/linux $ uname -a
Linux raspberrypi 6.12.62+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.12.62-1+rpt1 (2025-12-18) aarch64 GNU/Linux
```
Check uname after **reboot**:
```
pi05@raspberrypi:~ $ uname -a
Linux raspberrypi 6.12.62+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.12.62-1+rpt1 (2025-12-18) aarch64 GNU/Linux
```


