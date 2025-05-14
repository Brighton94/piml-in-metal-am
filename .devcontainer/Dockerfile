FROM python:3.12-bookworm

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Install dependencies and create the user
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 systemctl \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

WORKDIR /piml-in-metal-am

# Install PyTorch (with CUDA 12.8 support, depending on architecture) and ultralytics
RUN pip install --upgrade pip && \
    if [ "$TARGETARCH" = "amd64" ] ; then \
      pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision ultralytics ; \
    else \
      pip install torch torchvision ultralytics ; \
    fi

COPY src src
COPY pyproject.toml pyproject.toml
COPY README.md README.md
RUN pip install --upgrade pip && pip install .

USER $USERNAME

CMD ["tail", "-f", "/dev/null"]
