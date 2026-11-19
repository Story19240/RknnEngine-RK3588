# RknnEngine

RknnEngine 是面向 RK3588 Android 设备的通用 RKNN 推理引擎。客户端传入一帧 `Bitmap`，引擎完成模型加载、等比缩放与补边预处理、NPU 推理和原始张量输出；相机、业务算法、结果展示均不在本仓库范围内。

## 模块结构

```text
RknnEngine-RK3588/
├── android/                    # Android 推理引擎源码
│   └── rknn-engine/
│       └── src/
│           ├── main/
│           │   ├── assets/
│           │   ├── cpp/
│           │   └── java/com/rknn/engine/
│           └── test/
├── convert/                    # 独立的模型转换工具
│   ├── docker/                 # Docker 配置、依赖安装包
│   ├── scripts/                # Shell 入口脚本
│   ├── python/                 # Python 源码和依赖清单
│   ├── work/                   # 本地权重、数据集和转换产物，不入库
│   ├── check_rknn.sh           # 板端 RKNN 环境检查
│   ├── RK3588模型转换说明.md
│   └── README.md
├── docs/                       # 引擎接入文档
└── README.md
```

Java API 位于 `com.rknn.engine`：`RknnEngine` 提供 `loadModel`、`submitFrame` 和
`close`，通过 `EngineCallback` 返回 `InferenceResult`。通用加载方式继续透传
`TensorOutput[]`；传入 `DetectionConfig` 时，native 层会完成 YOLOv8 slim-head
解码，并通过 `InferenceResult.getDetectionResult()` 返回结构化检测框。

## 运行时口径

- 随包 RKNN runtime 为 **1.5.0**，适配 RK3588 板载 NPU 驱动 **v0.8.8**。
- 模型须使用 RKNN Toolkit2 **1.5.0** 转换，并指定 `target_platform=rk3588`。
- OpenCV Mobile 4.10.0、RKNN 头文件和 arm64-v8a 运行时位于 `third_party/`。
- ABI 仅支持 `arm64-v8a`；Gradle 8.7、AGP 8.5.2、JDK 17、NDK 26.1.10909125、CMake 3.22.1。

## 构建

```bash
cd android
./gradlew :rknn-engine:assembleDebug
```

## 模型转换

`convert/` 提供可复现的 YOLOv8 slim-head 转换流程，默认输出
`output_reg [1,64,8400]` 和 `output_cls [1,5,8400]`。Docker 环境使用
Toolkit2 1.5.0，与目标板 `librknnrt.so` 1.5.0 / NPU 驱动 0.8.8 对齐：

使用 Docker Compose 可以一次完成镜像构建和模型转换。准备
`convert/work/best.pt` 和 `convert/work/dataset/` 后，执行：

```bash
cd convert/docker
cp .env.example .env
docker compose run --rm --build converter
```

旧版 Compose 可先执行 `docker-compose build converter`，再执行
`docker-compose run --rm converter`。生成的 ONNX、RKNN、校准清单和 SHA-256
记录位于 `convert/work/output/`。权重、校准图片、输出目录和转换参数均可在
`convert/docker/.env` 中配置，不会打入镜像或提交到 Git。

完整的输入格式、FP16、优化级别和排障说明见
[`convert/README.md`](convert/README.md)。

导出的 slim ONNX 可用 Ultralytics 与仓库参考解码器做真实图片对照：

```bash
docker compose run --rm --entrypoint python3 converter \
  /opt/convert/verify_slim.py \
    --weights /input/best.pt --onnx /out/best_slim.onnx \
    --image /calib/class_a/example.jpg \
    --output /out/verify_slim_real_image.json
```

该命令验证 PC 端 float ONNX 与参考后处理，不执行 RKNN runtime，也不替代
RK3588 真机上的量化模型验证。INT8 opt=1 和 16bit opt=3 可复用同一转换命令，
分别修改 `--optimization-level` 和 `--quantized`。

详细过程、模型清单和真机验收边界见
[`convert/RK3588模型转换说明.md`](convert/RK3588模型转换说明.md)。

Android 工程引入、API、后台推理和生命周期管理见
[`docs/引擎接入指南.md`](docs/引擎接入指南.md)。

## 测试

```bash
./android/rknn-engine/src/test/cpp/run_host_tests.ps1
```

Android assets 中包含本次 5 类缺陷模型的两个 INT8 slim 文件：
`yolov8n.rknn` 对应 opt=3，`yolov8n_opt1.rknn` 对应 opt=1；源文件及
转换记录仍保留在 `convert/work/output/`。加载模型时应传入
`DetectionConfig.forDefectModel()`，实际项目也可根据自己的类别配置创建
`DetectionConfig`。
