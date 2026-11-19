# SDK assets

当前目录中的 `yolov8n.rknn` 已替换为本次 5 类缺陷模型的 INT8 opt=3 slim
候选模型。它由 `convert/work/output/best_slim_int8_opt3.rknn` 按运行侧约定改名复制而来，
文件名用于兼容现有 assets 加载方式，实际模型配置以转换 manifest 为准。

当前 assets 文件：

```text
yolov8n.rknn
  source : convert/work/output/best_slim_int8_opt3.rknn
  size   : 11,946,967 字节
  sha256 : eb98fb1ac7352fdc2e1d3b05bbb19a318764ed82efed88ba01a3b513e1a73470
  target: rk3588
  quant : INT8 / asymmetric_quantized-8
  opt   : 3
  output: output_reg [1,64,8400], output_cls [1,5,8400]

yolov8n_opt1.rknn
  source : convert/work/output/best_slim_int8_opt1.rknn
  size   : 11,962,839 字节
  sha256 : c9d9d1780aadb3bd26cdc5e08e4bda1d80b7e1a2aee9ff7fe78b4ef729fc038f
  target: rk3588
  quant : INT8 / asymmetric_quantized-8
  opt   : 1
  output: output_reg [1,64,8400], output_cls [1,5,8400]
```

`yolov8n.rknn` 是优先测试的主候选，`yolov8n_opt1.rknn` 是图优化级别较低的
备选模型。`best_slim_16bit_opt3.rknn` 继续只保留在 `convert/work/output/`，作为
量化精度基线，不放入当前 Android assets。

这两个模型尚未通过 RK3588 真机验收，因此仍是候选资产。接入时必须使用带
`DetectionConfig.forDefectModel()` 的加载方式，并同时保存所选模型的 SHA-256、
转换参数、输出形状和验收记录。
