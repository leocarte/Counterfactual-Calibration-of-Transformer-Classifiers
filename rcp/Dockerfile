FROM registry.rcp.epfl.ch/rcp-courses/2026-ee-559:latest

RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.12 python3.12-venv python3.12-dev && \
    rm -r /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Split heavy deps into separate RUN layers for resilient pushes
# (each layer uploads independently; partial-push failures don't lose the others).
# vllm is intentionally NOT installed here — it's ~10 GB (bundles CUDA kernels)
# and blocks push on normal networks. LLM jobs install it at runtime via submit_llm.sh.
RUN pip install "torch>=2.1.0"
RUN pip install "transformers>=4.36.0" "datasets>=2.16.0" "tokenizers>=0.15.0" \
                "accelerate>=0.25.0" "safetensors>=0.4.0" "sentencepiece>=0.2.0" \
                "tiktoken>=0.7.0" "bitsandbytes>=0.42.0"

# Rest of the stack (scikit-learn, pandas, omegaconf, wandb, matplotlib, pytest, etc.)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# NOTE: no WORKDIR — /home/case/antisemitism-detection comes from the home PVC,
# and setting WORKDIR to a PVC-mounted path caused runc to fail `mkdir` at boot
# with "permission denied" under UID 307974. We cd into the repo inside submit.sh's
# CONTAINER_CMD instead.
