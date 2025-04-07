# Development Container Setup

This directory contains configuration for a development container that provides a consistent Python environment for the project.

## Prerequisites

### For Linux Users
- Install Docker and Docker Compose:
  ```bash
  # Install Docker
  sudo apt-get update
  sudo apt-get install docker.io
  sudo systemctl enable --now docker
  
  # Add your user to the docker group (to avoid using sudo with docker)
  sudo usermod -aG docker $USER
  # Log out and back in for this to take effect
  
  # Install Docker Compose
  sudo apt-get install docker-compose
  ```

### For macOS Users
- If using Docker Desktop:
  - Install Docker Desktop from https://www.docker.com/products/docker-desktop
  
- If using Lima:
  - Install Lima and Docker:
    ```bash
    brew install lima docker docker-compose
    limactl start template://docker
    ```
  - For VS Code, you might need to set your DOCKER_HOST environment variable:
    ```bash
    export DOCKER_HOST=unix:///$HOME/.lima/docker/sock/docker.sock
    ```

## Using the Development Container

1. Open this project in VS Code or Cursor
2. If prompted to "Reopen in Container", click that option
3. Otherwise, click the green icon in the bottom-left corner and select "Reopen in Container"

## Troubleshooting

- If the container fails to build, make sure Docker is running
- For Linux users: ensure your user is in the docker group
- For macOS users with Lima: ensure your DOCKER_HOST environment variable is set correctly
- If file permissions are an issue, you may need to modify the volume mount options in docker-compose.yml 