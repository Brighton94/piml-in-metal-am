FROM python:3.12

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        sudo git curl ca-certificates build-essential \
        libssl-dev libffi-dev libpq-dev libsqlite3-dev unzip \
        libgl1 x11-apps libx11-dev python3-tk tk-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN if ! getent group ${USER_GID}; then groupadd --gid ${USER_GID} ${USERNAME}; fi \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN cp /etc/skel/.bashrc /home/${USERNAME}/.bashrc \
    && cp /etc/skel/.profile /home/${USERNAME}/.profile \
    && chsh -s /bin/bash ${USERNAME} \
    && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc /home/${USERNAME}/.profile

USER ${USERNAME}
WORKDIR /piml-in-metal-am

ENV PATH="/home/${USERNAME}/.local/bin:$PATH"
ENV DATASET_ROOT=/data

CMD [ "bash" ]
