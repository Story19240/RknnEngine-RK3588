FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        python3 \
        python3-pip \
        python3.10 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/convert

COPY python/requirements.txt /opt/convert/requirements.txt
RUN python3 -m pip install --upgrade "pip==24.0" \
    && python3 -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.4.0" "torchvision==0.19.0" \
    && python3 -m pip install -r /opt/convert/requirements.txt

COPY docker/rknn_toolkit2-1.5.0+1fa95b5c-cp310-cp310-linux_x86_64.whl /tmp/
RUN python3 -m pip install --no-deps \
        /tmp/rknn_toolkit2-1.5.0+1fa95b5c-cp310-cp310-linux_x86_64.whl \
    && rm -f \
        /tmp/rknn_toolkit2-1.5.0+1fa95b5c-cp310-cp310-linux_x86_64.whl

RUN python3 -c \
    "import importlib.metadata as m; import onnx, onnxruntime, torch, ultralytics; from rknn.api import RKNN; print('rknn-toolkit2=' + m.version('rknn-toolkit2'))"

COPY python/export_onnx.py \
     python/prepare_quant_dataset.py \
     python/to_rknn.py \
     python/decode_slim.py \
     python/verify_slim.py \
     scripts/run_convert.sh \
     scripts/entrypoint.sh \
     /opt/convert/

RUN sed -i 's/\r$//' /opt/convert/*.sh \
    && chmod +x /opt/convert/*.sh

WORKDIR /work
ENTRYPOINT ["/opt/convert/entrypoint.sh"]
CMD ["convert", "--help"]
