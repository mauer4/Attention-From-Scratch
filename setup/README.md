# Setup

This area captures everything required to bootstrap local or containerised
environments before working with the model artifacts.

- `venv/` - helper scripts or notes for creating/activating virtual
  environments.
- `requirements/` - dependency lists or lock files (used by the helper scripts).
- `docker/` - optional container definitions.

Helper scripts:

- `bare_metal_setup.sh` - installs apt-based prerequisites and bootstraps a
  Python virtual environment in one step.
- `venv/create_*.{ps1,sh,csh}` - shell-specific helpers for managing the venv
  when system packages are already present.

Run `venv/create_venv.ps1` to create and populate a Python virtual environment
using the requirements file, or replicate those commands manually on other
platforms.

The rest of the repository assumes these steps have been run first.

