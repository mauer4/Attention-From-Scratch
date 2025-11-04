# Development Environment Setup

This document provides detailed instructions for setting up the development environment for the Attention-From-Scratch project. There are two ways to set up the environment:

1.  **Using Docker (Recommended):** This is the easiest and most reliable way to get started. Docker ensures that you have a consistent and reproducible development environment.
2.  **Manual Setup:** If you prefer not to use Docker, you can set up the environment manually.

## Using Docker

To set up the development environment using Docker, follow these steps:

1.  **Install Docker:** If you don't have Docker installed, please follow the official instructions for your operating system:
    *   [Install Docker Engine](https://docs.docker.com/engine/install/)
    *   [Install Docker Compose](https://docs.docker.com/compose/install/)

2.  **Build the Docker image:**
    ```bash
    docker-compose build
    ```

3.  **Run the Docker container:**
    ```bash
    docker-compose run --rm dev
    ```

4.  **Inside the container, run the setup script:**
    ```bash
    bash scripts/setup/install_all.sh
    ```

## Manual Setup

If you prefer to set up the environment manually, you will need to have the following prerequisites installed on your system:

*   **NVIDIA GPU with CUDA:** An NVIDIA GPU with CUDA installed is required to run the project.
*   **Python 3.8 or higher:** You will need to have Python 3.8 or a higher version installed on your system.
*   **pip:** You will need to have `pip` installed to manage the Python dependencies.

Once you have the prerequisites installed, you can follow these steps to set up the environment:

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```

2.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Install the Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the setup script:**
    ```bash
    bash scripts/setup/install_all.sh
    ```
