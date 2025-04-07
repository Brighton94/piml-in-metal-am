FROM python:3.12

# Use ARGs for user config
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        sudo \
        git \
        curl \
        ca-certificates \
        build-essential \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        libsqlite3-dev \
        unzip \
        libgl1 \
        x11-apps \
        libx11-dev \
        python3-tk \
        tk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN cp /etc/skel/.bashrc /home/${USERNAME}/.bashrc \
    && cp /etc/skel/.profile /home/${USERNAME}/.profile \
    && chsh -s /bin/bash ${USERNAME} \
    && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc /home/${USERNAME}/.profile


# Switch to non-root user
USER ${USERNAME}
WORKDIR /piml-in-metal-am

# Install Python dependencies (editable install expects pyproject.toml or setup.py to be present)
RUN pip install --upgrade pip \
    && pip install -e ".[dev]" || true \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Add local Python binaries to PATH
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"
ENV DATASET_ROOT=/data

CMD [ "bash" ]
