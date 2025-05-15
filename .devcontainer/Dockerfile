FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

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
USER $USERNAME
WORKDIR /piml-in-metal-am

COPY pyproject.toml README.md ./
RUN pip install --user --upgrade pip && pip install --user -e ".[dev]"
COPY src src

CMD ["tail", "-f", "/dev/null"]
