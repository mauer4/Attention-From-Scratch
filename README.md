# Attention-From-Scratch

Attention-From-Scratch is a project dedicated to building a custom AI inference engine from the ground up, using the OLMo 2 open-source LLM weights. The project also supports running the Allen AI OLMo 2 inference engine for performance comparison.

## Quick Start

There are two ways to set up the development environment for this project:

1.  **Using Docker (Recommended):** This is the easiest and most reliable way to get started. Docker ensures that you have a consistent and reproducible development environment.
2.  **Manual Setup:** If you prefer not to use Docker, you can set up the environment manually by following the instructions in the `docs/SETUP.md` file.

### Using Docker

To set up the development environment using Docker, follow these steps:

1.  **Build the Docker image:**
    ```bash
    docker-compose build
    ```
2.  **Run the Docker container:**
    ```bash
    docker-compose run --rm dev
    ```
3.  **Inside the container, run the setup script:**
    ```bash
    bash scripts/setup/install_all.sh
    ```

### Manual Setup

For manual setup instructions, please refer to the `docs/SETUP.md` file.

## Repository Layout

-   `src/`: The source code for the project, including the custom inference engine and the C++ code.
-   `scripts/`: Scripts for setting up the environment, running inference, and performing analysis.
-   `config/`: Configuration files for the project.
-   `weights/`: Directory for storing the model weights.
-   `reports/`: Generated reports, such as benchmark results and environment summaries.
-   `docs/`: Project documentation.

## Custom Engine Progress

The custom inference engine is currently under development. The goal is to implement custom CUDA kernels and a bespoke runtime graph to optimize performance. You can track the progress and milestones in the `docs/ARCHITECTURE.md` and `docs/PROJECT_PLAN.md` files.
