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

# Install PyTorch system-wide (before switching to user)
RUN pip install --upgrade pip && \
    pip install torch torchvision

WORKDIR /piml-in-metal-am
COPY src src
COPY pyproject.toml pyproject.toml
COPY README.md README.md
RUN pip install --upgrade pip && pip install .

USER $USERNAME

CMD ["tail", "-f", "/dev/null"]
