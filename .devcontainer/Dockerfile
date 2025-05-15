FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

# Install dependencies and create the user
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1 \
    python3-tk \
    systemctl \
    x11-apps \
    x11-utils \
    xauth \
    && groupadd --gid "$USER_GID" "$USERNAME" \
    && useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME" \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -e ".[dev]"

USER $USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
WORKDIR /piml-in-metal-am

COPY src src
CMD ["tail", "-f", "/dev/null"]
