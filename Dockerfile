ARG PYTHON_BASE=2.4.0-cuda12.4-cudnn9-devel
#ARG PYTHON_BASE=2.6.0-cuda12.6-cudnn9-devel

# build stage
FROM pytorch/pytorch:$PYTHON_BASE AS builder

RUN apt update && apt upgrade -y
RUN apt install vim git -y

RUN DEBIAN_FRONTEND="noninteractive" TZ="Europe/Oslo" apt install portaudio19-dev ffmpeg chromium-codecs-ffmpeg-extra build-essential -y
RUN apt install libmagic1 wget libnvinfer10 libnvonnxparsers10  tensorrt tensorrt-libs tensorrt-dev python3-pip -y

RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    apt install -y build-essential \
    git \
    clang \
    libnvinfer-lean10 libnvinfer-vc-plugin10 python3-libnvinfer-lean poppler-utils tesseract-ocr-all

RUN mkdir -p /root/.paddleocr
#COPY Packages/paddleocr /root/.paddleocr
RUN if [ ! -f /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer.tar ]; \
    then \
    echo -e "\033[0;31m\u10102 - en_PP-OCRv3_det_infer.tar NOT found! Downloading....!\033[0m" \
    mkdir -p /root/.paddleocr/whl/det ; \
    wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -P /root/.paddleocr/whl/det/en/ \
    else ; \
    echo -e "\033[0;32m\u2705 - en_PP-OCRv3_det_infer.tar found! Skipping download!\033[0m" ; \
    fi
WORKDIR /root/.paddleocr/whl/det/en
RUN tar xvf en_PP-OCRv3_det_infer.tar

RUN if [ ! -f /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer.tar ]; \
    then \
    echo -e "\033[0;31m\u10102 - en_PP-OCRv3_rec_infer.tar NOT found! Downloading....!\033[0m" \
    mkdir -p /root/.paddleocr/whl/rec/en ; \
    wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar -P /root/.paddleocr/whl/rec/en/ \
    else ; \
    echo -e "\033[0;32m\u2705 - en_PP-OCRv3_rec_infer.tar found! Skipping download!\033[0m" ; \
    fi
WORKDIR /root/.paddleocr/whl/rec/en
RUN tar xvf en_PP-OCRv3_rec_infer.tar

RUN if [ ! -f /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer.tar ]; \
    then \
    echo -e "\033[0;31m\u10102 - ch_ppocr_mobile_v2.0_cls_infer.tar NOT found! Downloading....!\033[0m" \
    mkdir -p /root/.paddleocr/whl/cls ; \
    wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -P /root/.paddleocr/whl/cls/ \
    else ; \
    echo -e "\033[0;32m\u2705 - ch_ppocr_mobile_v2.0_cls_infer.tar found! Skipping download!\033[0m" ; \
    fi
WORKDIR /root/.paddleocr/whl/cls
RUN tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar


# install PDM
RUN pip install -U pdm

# disable update check
ENV PDM_CHECK_UPDATE=false
# copy files
RUN mkdir -p /project
COPY pyproject.toml pdm.lock README.md /project/
COPY src/ /project/

WORKDIR /opt/conda/lib/python3.11/site-packages/torch/lib
RUN ln -s libcudnn_adv.so.9 libcudnn_adv.so \
  && ln -s libcudnn.so.9 libcudnn.so \
  && ln -s libcudnn_cnn.so.9 libcudnn_cnn.so \
  && ln -s libcudnn_engines_precompiled.so.9 libcudnn_engines_precompiled.so \
  && ln -s libcudnn_engines_runtime_compiled.so.9 libcudnn_engines_runtime_compiled.so \
  && ln -s libcudnn_graph.so.9 libcudnn_graph.so \
  && ln -s libcudnn_heuristic.so.9 libcudnn_heuristic.so \
  && ln -s libcudnn_ops.so.9 libcudnn_ops.so

ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# install dependencies and project into the local packages directory
WORKDIR /project
RUN --mount=type=cache,target=/root/.cache/pip pdm install --check --prod --no-editable \
    && pdm run python -m ensurepip

ENV PATH="/project/.venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip pdm run python -m pip uninstall onnxruntime -y \
    && pdm run python -m pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

RUN pdm run python -m pip freeze | grep onnx

# run stage
#FROM pytorch/pytorch:$PYTHON_BASE

# retrieve packages from build stage
#COPY --from=builder /root/.local/ /root/.local
#COPY --from=builder /project/.venv/ /project/.venv

#ENV PATH="/project/.venv/bin:$PATH"
ENV PYTHONPATH="/project/.venv/bin:/project:/app"

EXPOSE 7860

# set command/entrypoint, adapt to fit your needs
CMD ["python3", "main.py"]
