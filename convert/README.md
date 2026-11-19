# RKNN 模型转换工具

本目录提供通用的 YOLOv8 → ONNX → RKNN 转换示例。Docker 中包含 PyTorch、
Ultralytics、ONNX Runtime 和 RKNN Toolkit2 等依赖，权重、校准图片及生成物
均保存在本地工作区，不会打入镜像。

正确的使用顺序是：**先检查开发板环境，再构建转换环境，最后转换模型**。

## 一、先检查开发板 RKNN 环境

在转换模型之前，先运行 `check_rknn.sh`，确认开发板上的用户态库、NPU 驱动
和 RKNN 服务是否符合部署要求。不要等到模型转换完成后才发现版本不匹配。

### 1. 需要检查的对象

| 名称 | 类型 | 作用 |
|---|---|---|
| `librknnrt.so` | 当前 RKNN 用户态运行库 | 通过 RKNN Runtime API 加载模型并执行推理 |
| `librknn_api.so` | 旧版 RKNN 用户态 API 库 | 为仍包含该库的旧系统提供 RKNN API |
| `rknpu` | NPU 内核驱动 | 负责与 NPU 硬件通信，不是动态库 |
| `rknn_server` | RKNN 服务可执行文件/进程 | 提供用户态 RKNN 服务，不是动态库 |

重点核对 `librknnrt.so` 的版本，并确认它与转换模型所用的 RKNN Toolkit2
及 NPU 驱动兼容。本示例默认使用 Toolkit2 1.5.0，其他版本需要单独核对。

`librknnrt.so` 负责解析模型、提供 RKNN API、管理输入输出和上下文，并组织
推理任务后提交给底层。`rknpu` 驱动负责内存与任务调度、硬件中断、超时和
复位，最终控制 NPU 执行。只升级动态库并不能提升旧驱动的硬件兼容能力。

### 2. Toolkit2、动态库与驱动兼容参考

| RKNN Toolkit2 | `librknnrt.so` | RKNPU 内核驱动 | 依据与说明 |
|---|---|---|---|
| `1.5.0` | `1.5.0` | `0.8.8` | 本示例使用的组合，不表示最低版本限制 |
| `1.6.0` | `1.6.0` | `>= 0.9.2` | 对应版本官方快速启动指南 |
| `2.0.0-beta0` | `2.0.0-beta0` | `>= 0.9.2` | 对应版本官方快速启动指南 |
| `2.1.0` | `2.1.0` | `>= 0.9.2` | 对应版本官方快速启动指南 |
| `2.2.0` | `2.2.0` | `>= 0.9.2` | 对应版本官方快速启动指南 |
| `2.3.0` | `2.3.0` | `>= 0.9.2` | 对应版本官方快速启动指南 |
| `2.3.2` | `2.3.2` | `>= 0.9.2` | 官方快速启动指南；本检查脚本采用更严格的 `>= 0.9.8` 作为目标 |

**RKNN Toolkit2 下载地址：**
[Rockchip 官方发布页](https://github.com/airockchip/rknn-toolkit2/releases)。
请在发布页选择需要转换环境使用的版本，本示例使用 `1.5.0`。

`check_rknn.sh` 对 `2.3.2` 使用 `>= 0.9.8`，是为了给新版部署提供更严格的
检查目标，并不表示官方快速启动指南中的 `>= 0.9.2` 写错了。实际项目应根据
具体芯片、模型和 SDK 发布说明选择版本，不能认为满足表中的最低值就能运行
所有模型。

版本信息可参考 [Rockchip RKNN Toolkit2 发布页](https://github.com/airockchip/rknn-toolkit2/releases)
及各版本随附的快速启动指南。例如：
[RKNN SDK 2.3.2 快速启动指南](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/doc/01_Rockchip_RKNPU_Quick_Start_RKNN_SDK_V2.3.2_CN.pdf)。

实际部署时，**Toolkit2、`librknnrt.so` 和 RKNPU 驱动需要同时匹配**：

- Toolkit2 决定模型格式、算子和任务描述。
- `librknnrt.so` 负责理解模型并通过 API 组织推理。
- RKNPU 驱动负责将任务提交给硬件执行。

模型能够生成或成功调用 `rknn_init`，不代表旧驱动一定能执行 `rknn_run`。

### 3. 驱动版本高也不能随意替换运行库

NPU 驱动版本高，通常意味着可选择的运行库范围更大，但这只是必要条件，
不代表可以直接换成任意高版本的 `librknnrt.so`。还需要同时确认：

1. Toolkit2 生成的模型格式和任务描述能够被运行库正确解析。
2. `librknnrt.so` 与 `rknn_server` 的版本和接口匹配，建议使用同一套 SDK。
3. 运行库与系统的 CPU 架构和用户态环境匹配，例如不能混用 `arm64-v8a`、
   32 位 ARM 或不同平台的库。
4. 编译应用使用的 RKNN 头文件与运行库尽量来自同一套 SDK。

正确的升级顺序是：

```text
确认 NPU 驱动版本
    ↓
选择驱动支持的 RKNPU2 SDK 版本
    ↓
成套更新 librknnrt.so、rknn_server 和相关头文件
    ↓
使用对应 Toolkit2 转换模型
    ↓
验证 rknn_init、rknn_run、输出结果和稳定性
```

升级前应备份原库，确保出现问题时可以回退。不能只替换一个最高版本的
`librknnrt.so`，就认为旧模型和新模型都能正常运行。

### 4. 执行检查脚本

在仓库根目录执行：

```bash
adb push convert/check_rknn.sh /data/local/tmp/check_rknn.sh
adb shell sh /data/local/tmp/check_rknn.sh
```

建议使用 root 权限，以便读取完整的驱动和 debugfs 信息。

脚本不会替换任何库。如果 `/data/local/tmp/librknnrt.so` 存在，脚本还会
显示它的版本和校验值，但不会安装该文件。

## 二、目录说明

```text
convert/
├── docker/                    # Dockerfile、Compose、环境模板和 Toolkit2 wheel
├── scripts/                   # Shell 入口脚本
├── python/                    # Python 源码和依赖清单
├── work/                      # 本地权重、数据集和生成物，不入库
├── check_rknn.sh              # 板端 RKNN 环境检查脚本
├── RK3588模型转换说明.md       # 转换原理和详细示例
└── README.md                  # 使用说明
```

`work/` 中除说明文件外，其余内容均不进入 Git 和 Docker 构建上下文。

## 三、转换环境要求

- x86_64 Linux 主机，或使用 Linux 容器的 Windows/macOS Docker Desktop。
- Docker Engine 和 Docker Compose，支持新版 `docker compose` 或旧版
  `docker-compose`。
- 首次构建镜像时需要联网。
- 建议至少预留 8 GB 磁盘空间。

该环境在 PC 上执行模型转换，不负责在 RK3588 上运行推理，也不需要
NVIDIA GPU。

## 四、准备权重和校准图片

默认输入目录如下，文件名和类别目录均可按实际情况调整：

```text
convert/work/
├── best.pt
├── dataset/
│   ├── class_a/
│   │   └── *.jpg
│   ├── class_b/
│   │   └── *.jpg
│   └── ...
└── output/
```

`best.pt` 是 YOLOv8 权重示例。`dataset/` 用于 INT8 量化校准，一级子目录
用于按类均衡抽样。

FP16 转换不执行 INT8 校准，因此不需要校准图片。

## 五、构建转换环境

### 1. 创建本地配置

```bash
cd convert/docker
cp .env.example .env
```

Windows PowerShell 可以使用：

```powershell
Copy-Item .env.example .env
```

### 2. 构建镜像

```bash
docker compose build converter
```

旧版 Compose 使用：

```bash
docker-compose build converter
```

首次构建需要下载依赖，后续会复用 Docker 缓存。默认镜像名为
`rknn-convert:1.5.0-slim`。

### 3. 验证环境

```bash
docker compose run --rm --entrypoint python3 converter -c "import torch, onnx, onnxruntime, ultralytics; from rknn.api import RKNN; print('RKNN conversion environment OK')"
```

看到以下输出即表示关键依赖可以正常加载：

```text
RKNN conversion environment OK
```

## 六、执行模型转换

在 `convert/docker/` 目录执行：

```bash
docker compose run --rm converter
```

旧版 Compose 使用：

```bash
docker-compose run --rm converter
```

如果修改了 Dockerfile、依赖或转换脚本，可以在转换时同时重新构建：

```bash
docker compose run --rm --build converter
```

临时容器在转换结束后自动删除，生成文件保留在 `convert/work/output/`：

```text
convert/work/output/best_slim.onnx
convert/work/output/best_slim.onnx.json
convert/work/output/best_slim_dataset.txt
convert/work/output/best_slim_calibration.json
convert/work/output/best_slim_int8_opt3.rknn
convert/work/output/best_slim_int8_opt3.rknn.json
```

相同配置的重复转换会覆盖同名输出。需要保留历史结果时，请在 `.env` 中
修改 `OUTPUT_DIR`。

## 七、配置参数

编辑 `convert/docker/.env` 修改配置：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `WEIGHTS_DIR` | `../work` | 权重所在目录 |
| `WEIGHTS_FILE` | `best.pt` | 权重文件名 |
| `CALIB_DIR` | `../work/dataset` | 校准图片目录 |
| `CALIB_NUM` | `200` | 校准图片总数 |
| `OUTPUT_DIR` | `../work/output` | 输出目录 |
| `IMAGE_NAME` | `rknn-convert:1.5.0-slim` | Docker 镜像名 |
| `PLATFORM` | `rk3588` | RKNN 目标平台 |
| `INPUT_SIZE` | `640` | 正方形输入边长 |
| `QUANTIZED` | `INT8` | `INT8` 或 `FP16` |
| `HEAD_MODE` | `slim` | `slim` 或兼容对照模式 `fused` |
| `OPTIMIZATION_LEVEL` | `3` | RKNN 图优化级别 `1/2/3` |
| `OPSET` | `12` | ONNX opset |
| `SEED` | `42` | 校准抽样随机种子 |

相对路径以 `convert/docker/` 为基准。默认的 `../work`、`../work/dataset`
和 `../work/output` 分别对应 `convert/work/` 下的同名目录。

使用 FP16 时设置：

```dotenv
QUANTIZED=FP16
```

此时不需要校准图片，默认输出名为 `best_slim_16bit_opt3.rknn`。

## 八、转换结果检查与验证

### 1. 检查输出清单

转换脚本成功后，先确认 `.rknn` 和同名 `.rknn.json` 同时存在。清单重点字段：

| 字段 | 用途 |
|---|---|
| `source_onnx_sha256` | 确认 RKNN 来源 ONNX |
| `dataset_sha256` | 确认 INT8 使用的校准清单 |
| `output_rknn_sha256` | 核对最终模型文件是否被替换或损坏 |
| `target_platform` | 确认目标平台，例如 `rk3588` |
| `quantized_dtype` | 查看实际量化类型 |
| `optimization_level` | 查看图优化级别 |
| `toolkit2_version` | 核对转换工具版本 |

复制模型到 Android 工程或开发板后，建议重新计算 SHA-256，并与清单中的
`output_rknn_sha256` 对比。

### 2. 进行 PC 侧数值验证

以下命令演示如何对照 Ultralytics 与参考解码器。请先将示例权重、ONNX 和
图片路径替换为实际文件：

```bash
cd convert/docker
docker compose run --rm --entrypoint python3 converter \
  /opt/convert/verify_slim.py \
    --weights /input/best.pt \
    --onnx /out/best_slim.onnx \
    --image /calib/class_a/example.jpg \
    --output /out/verify_slim.json
```

该检查用于发现 DFL、anchor、stride、NMS 或坐标反变换错误，不能替代板端
精度和稳定性验证。

### 3. 进行板端验证

将模型部署到 Android 后，至少检查：

- `rknn_init` 能成功加载模型。
- 实际输入输出名称、形状、类型和布局与转换清单一致。
- `rknn_run` 连续执行时没有超时、任务提交失败或 NPU reset。
- 检测结果与 PC 参考结果之间的差异在业务允许范围内。
- 横图、竖图、空目标、多目标和贴边目标的坐标均正确。
- 连续运行时的内存、温度、推理耗时和稳定性满足要求。

## 九、不同转换需求的配置示例

### 1. INT8 生产候选模型

```dotenv
QUANTIZED=INT8
HEAD_MODE=slim
OPTIMIZATION_LEVEL=3
CALIB_NUM=200
```

INT8 需要校准图片。校准图片应覆盖实际使用场景，不要使用与业务无关的
公开图片简单凑数。

### 2. FP16 精度对照模型

```dotenv
QUANTIZED=FP16
HEAD_MODE=slim
OPTIMIZATION_LEVEL=3
```

FP16 不读取校准图片，可用于对照 INT8 的量化误差。默认输出名为
`best_slim_16bit_opt3.rknn`。

### 3. 图优化级别对照

保持权重、校准集、输入尺寸和后处理参数不变，分别设置：

```dotenv
OPTIMIZATION_LEVEL=1
```

和：

```dotenv
OPTIMIZATION_LEVEL=3
```

比较两个模型的加载、推理、精度和稳定性。优化级别更高不保证在所有驱动和
模型上都更合适，不能只根据文件大小或 PC 转换日志选择模型。

### 4. 自定义校准数量

默认在一级类别目录之间均分 `CALIB_NUM`。例如 4 个目录、总数 200，则每个
目录抽取 50 张。总数不能整除时，应调整 `CALIB_NUM`，或直接调用转换入口
使用 `--per-class N` 明确指定每类数量。

## 十、常用命令

只构建或重新构建镜像：

```bash
docker compose build converter
```

复用已有镜像执行转换：

```bash
docker compose run --rm converter
```

进入转换环境的交互式 Shell：

```bash
docker compose run --rm --entrypoint bash converter -c "exec bash"
```

## 十一、常见问题

| 现象 | 检查方向 |
|---|---|
| 找不到 `best.pt` | 检查 `WEIGHTS_DIR`、`WEIGHTS_FILE` 和实际文件路径 |
| 找不到校准目录 | 检查 `CALIB_DIR`；FP16 不需要校准图片 |
| 校准数量无法均分 | 确保各类图片数量充足，且 `CALIB_NUM` 能被类别目录数整除 |
| wheel 或架构不受支持 | 使用 x86_64 Docker 主机，Toolkit2 1.5.0 不是 ARM Linux 安装包 |
| 构建时下载依赖失败 | 检查网络和 DNS 后重试，已完成的 Docker 层会使用缓存 |
| 输出目录无法写入 | 检查 `OUTPUT_DIR` 是否存在以及 Docker 是否有写权限 |
| 容器转换过程中退出 | 查看终端日志，或进入交互式 Shell 分步排查 |
| 模型在 PC 生成成功但板端无法运行 | 重新检查第一节中的库、驱动、Toolkit2 和目标平台兼容性 |

生成的 `.rknn.json` 记录源文件校验值、目标平台、量化类型、优化级别、
Toolkit2 版本和输出 SHA-256。建议将清单与模型一起保存，便于追溯。

转换原理、slim-head 输出和验证方法见
[`RK3588模型转换说明.md`](RK3588模型转换说明.md)。

Android 工程引入、模型放置、后台推理和生命周期管理见
[`docs/引擎接入指南.md`](../docs/引擎接入指南.md)。
