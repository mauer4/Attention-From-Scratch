# Setup

The primary entrypoint lives at `setup/bootstrap_host.sh`. It detects your
package manager, provisions CUDA when required, creates (or reuses) the target
virtual environment, installs the locked dependencies from
`requirements/locks/`, and logs detailed output to `logs/bootstrap_<timestamp>.log`.

Usage examples from the repository root:

```bash
bash setup/bootstrap_host.sh            # Full flow (system packages + CUDA + venv)
SKIP_SYSTEM_PACKAGES=1 bash setup/bootstrap_host.sh --python-env ~/.venvs/olmo
```

The script creates the environment but does **not** activate it for you; follow
the final line of the log (`source <path>/bin/activate`) in new shells.

Windows users should run the same commands inside WSL2. For additional options
(`--cuda-toolkit`, `--lock`, `--with-cutlass`, etc.), run
`bash setup/bootstrap_host.sh --help`.

