#!/system/bin/sh
# 用法（root）: sh check_rknn.sh
# 只读检查，不修改任何东西

ver() { grep -a -o "$2 version: [^)]*)" "$1" 2>/dev/null | head -n1; }

echo "===== 1. NPU 驱动 ====="
mount -t debugfs none /sys/kernel/debug 2>/dev/null
if [ -r /sys/kernel/debug/rknpu/version ]; then
    echo "debugfs: $(cat /sys/kernel/debug/rknpu/version)"
else
    echo "debugfs: 读不到 /sys/kernel/debug/rknpu/version"
fi
[ -r /sys/module/rknpu/version ] && echo "module version: $(cat /sys/module/rknpu/version)"
echo "-- dmesg:"
dmesg 2>/dev/null | grep -i rknpu | tail -n 5
echo "-- lsmod:"
lsmod 2>/dev/null | grep -i rknpu || echo "(rknpu 不是模块，或已编进内核)"
echo "-- 驱动模块文件:"
find /vendor /system /odm /vendor_dlkm /system_dlkm -name 'rknpu*' 2>/dev/null

echo
echo "===== 2. librknnrt / librknn_api ====="
find /vendor /system /system_ext /odm /data/local/tmp /data/app -type f \
    \( -name 'librknnrt.so' -o -name 'librknn_api.so' \) 2>/dev/null | while read f; do
    echo "$f"
    echo "    $(ver "$f" librknnrt)"
    echo "    md5: $(md5sum < "$f" | cut -d' ' -f1)  size: $(wc -c < "$f")"
done

echo
echo "===== 3. rknn_server ====="
find /vendor /system /system_ext /odm /data/local/tmp -type f -name 'rknn_server' 2>/dev/null | while read f; do
    echo "$f"
    echo "    $(ver "$f" rknn_server)"
done
echo "-- 运行中:"
ps -A 2>/dev/null | grep -i rknn || echo "(没有 rknn 相关进程)"

echo
echo "===== 4. 本次要换的新库 (如已 push) ====="
NEW=/data/local/tmp/librknnrt.so
if [ -f "$NEW" ]; then
    echo "$NEW"
    echo "    $(ver "$NEW" librknnrt)"
    echo "    md5: $(md5sum < "$NEW" | cut -d' ' -f1)"
else
    echo "未找到 $NEW"
fi

echo
echo "===== 完成 ====="
echo "对比: 目标 2.3.2，驱动需 >= 0.9.8"
