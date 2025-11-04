#!/usr/bin/env python3
"""
GPU and CUDA environment verification helper.

This script inspects the host for NVIDIA GPUs, installed drivers, CUDA runtime
tooling, and optional PyTorch backends (cuDNN/NCCL). Results are written to
`reports/system_gpu.json` and the detected CUDA version is echoed so callers can
export it (e.g. via `export $(python setup_env/check_gpu.py --emit-env)`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "system_gpu.json"
AUTODETECTED_ENV = ROOT / "config/config.yaml"


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"


def parse_nvidia_smi_summary(output: str) -> Tuple[Optional[str], Optional[str]]:
    driver_match = re.search(r"Driver Version:\s*([\w\.]+)", output)
    cuda_match = re.search(r"CUDA Version:\s*([\w\.]+)", output)
    driver = driver_match.group(1) if driver_match else None
    cuda = cuda_match.group(1) if cuda_match else None
    return driver, cuda


def detect_gpu() -> Dict[str, Any]:
    gpu_info: Dict[str, Any] = {
        "present": False,
        "details": [],
        "warnings": [],
    }

    rc, out, err = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,cuda_version",
            "--format=csv,noheader",
        ]
    )
    if rc != 0:
        fallback_rc, fallback_out, fallback_err = run_command(["nvidia-smi"])
        list_rc, list_out, _ = run_command(["nvidia-smi", "-L"])
        if fallback_rc == 0:
            driver, cuda = parse_nvidia_smi_summary(fallback_out)
        else:
            driver = cuda = None

        names = []
        if list_rc == 0:
            for line in list_out.splitlines():
                if line.startswith("GPU"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        names.append(parts[1].split("(")[0].strip())

        if names:
            gpu_info["present"] = True
            for idx, name in enumerate(names):
                gpu_info["details"].append(
                    {
                        "name": name,
                        "driver_version": driver,
                        "cuda_version": cuda,
                        "source": "nvidia-smi fallback",
                    }
                )
        else:
            gpu_info["warnings"].append(
                "Unable to query NVIDIA GPUs via nvidia-smi --query; fallback also failed."
            )
            if err:
                gpu_info["warnings"].append(err)
            if fallback_err:
                gpu_info["warnings"].append(fallback_err)
        return gpu_info

    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        gpu_info["warnings"].append("nvidia-smi returned no GPU entries.")
        return gpu_info

    gpu_info["present"] = True
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            name, driver_version, cuda_version = parts
            gpu_info["details"].append(
                {
                    "name": name,
                    "driver_version": driver_version,
                    "cuda_version": cuda_version,
                }
            )
        else:
            gpu_info["warnings"].append(
                f"Unexpected nvidia-smi output format: {line}"
            )
    return gpu_info


def detect_nvcc() -> Dict[str, Any]:
    rc, out, err = run_command(["nvcc", "--version"])
    info = {
        "available": rc == 0,
        "version_output": out if rc == 0 else "",
        "warnings": [],
    }
    if rc != 0:
        warn = "nvcc not found in PATH. Install CUDA toolkit if compilation tooling is required."
        info["warnings"].append(warn)
        if err:
            info["warnings"].append(err)
    return info


def detect_cudart() -> Dict[str, Any]:
    rc, out, err = run_command(["bash", "-lc", "ldconfig -p | grep cudart"])
    info = {
        "available": rc == 0 and bool(out.strip()),
        "entries": out.splitlines() if rc == 0 else [],
        "warnings": [],
    }
    if rc != 0 or not info["entries"]:
        warn = "Unable to find cudart via ldconfig. Ensure CUDA runtime libraries are on the linker path."
        info["warnings"].append(warn)
        if err:
            info["warnings"].append(err)
    return info


def detect_torch_backends() -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "torch_installed": None,
        "cudnn_version": None,
        "nccl_version": None,
        "warnings": [],
    }
    try:
        import torch  # type: ignore

        info["torch_installed"] = torch.__version__
        cudnn_ver = torch.backends.cudnn.version()
        info["cudnn_version"] = str(cudnn_ver) if cudnn_ver else None
        nccl_ver = (
            torch.cuda.nccl.version()
            if hasattr(torch.cuda, "nccl") and torch.cuda.is_available()
            else None
        )
        info["nccl_version"] = str(nccl_ver) if nccl_ver else None
    except ModuleNotFoundError:
        info["warnings"].append(
            "torch not installed; cuDNN and NCCL checks skipped."
        )
    except Exception as exc:  # pragma: no cover - safety net
        info["warnings"].append(f"Unexpected torch backend error: {exc}")
    return info


def derive_cuda_version(gpu_info: Dict[str, Any]) -> Optional[str]:
    if not gpu_info["present"]:
        return None
    cuda_versions = {
        entry.get("cuda_version", "").strip()
        for entry in gpu_info.get("details", [])
        if entry.get("cuda_version")
    }
    cuda_versions.discard("")
    if not cuda_versions:
        return None
    if len(cuda_versions) > 1:
        return sorted(cuda_versions)[-1]
    return cuda_versions.pop()


def build_report() -> Dict[str, Any]:
    gpu_info = detect_gpu()
    nvcc_info = detect_nvcc()
    cudart_info = detect_cudart()
    torch_info = detect_torch_backends()

    cuda_version = derive_cuda_version(gpu_info)
    driver_version = None
    if gpu_info["present"] and gpu_info["details"]:
        driver_versions = {
            d.get("driver_version") for d in gpu_info["details"] if d.get("driver_version")
        }
        driver_versions.discard(None)
        driver_version = sorted(driver_versions)[-1] if driver_versions else None

    report: Dict[str, Any] = {
        "gpu": gpu_info,
        "nvcc": nvcc_info,
        "cudart": cudart_info,
        "torch": torch_info,
        "detected_cuda_version": cuda_version,
        "detected_driver_version": driver_version,
        "warnings": [],
    }

    if not gpu_info["present"]:
        report["warnings"].append(
            "No NVIDIA GPU detected. Install an NVIDIA GPU or ensure the device is visible inside the container."
        )
    if cuda_version is None:
        report["warnings"].append(
            "CUDA version could not be detected via nvidia-smi."
        )
    if nvcc_info["available"] is False:
        report["warnings"].append(
            "nvcc not available. Install the CUDA toolkit if compilation is required."
        )
    if cudart_info["available"] is False:
        report["warnings"].append(
            "CUDA runtime libraries (cudart) missing from ldconfig cache."
        )

    return report


def emit_report(report: Dict[str, Any]) -> None:
    import yaml
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    cuda_version = report.get("detected_cuda_version") or ""

    config = {}
    if AUTODETECTED_ENV.exists():
        with open(AUTODETECTED_ENV, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    config["detected_cuda_version"] = cuda_version

    with open(AUTODETECTED_ENV, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    details = report.get("gpu", {}).get("details", [])
    if details:
        info = details[0]
        label = (
            f"{info.get('name', 'unknown')} | Driver {info.get('driver_version', 'unknown')} | "
            f"CUDA {info.get('cuda_version', cuda_version or 'unknown')}"
        )
        print(f"✅ GPU: {label}")
    else:
        print("⚠️  GPU: not detected via nvidia-smi")

    nvcc_available = report.get("nvcc", {}).get("available")
    print("✅ nvcc detected" if nvcc_available else "⚠️  nvcc missing (install CUDA toolkit)")

    cudart_available = report.get("cudart", {}).get("available")
    print("✅ cudart runtime located" if cudart_available else "⚠️  cudart runtime missing from ldconfig")

    torch_info = report.get("torch", {})
    if torch_info.get("torch_installed"):
        print(
            "✅ torch {torch} | cuDNN {cudnn} | NCCL {nccl}".format(
                torch=torch_info.get("torch_installed"),
                cudnn=torch_info.get("cudnn_version") or "?",
                nccl=torch_info.get("nccl_version") or "?",
            )
        )
    else:
        print("⚠️  torch not installed; backend checks skipped")

    for warning in report.get("warnings", []):
        print(f"⚠️  {warning}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify GPU / CUDA stack.")
    parser.add_argument(
        "--emit-env",
        action="store_true",
        help="Print `CUDA_VERSION=<value>` and exit.",
    )
    args = parser.parse_args(argv)

    report = build_report()
    cuda_version = report.get("detected_cuda_version")

    if args.emit_env:
        if cuda_version:
            env_line = f"CUDA_VERSION={cuda_version}"
            # Set it for current process to support chaining.
            os.environ["CUDA_VERSION"] = cuda_version
            print(env_line)
            return 0
        print("CUDA_VERSION=", end="")
        return 0

    if cuda_version:
        os.environ["CUDA_VERSION"] = cuda_version

    emit_report(report)
    print(f"✅ Report written to {REPORT_PATH.relative_to(ROOT)}")
    print(f"✅ Autodetected CUDA version recorded in {AUTODETECTED_ENV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
