#!/usr/bin/env python3
"""Aggregate environment verification artefacts into markdown and HTML reports."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import sys


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_env import get_model_identifiers, get_model_paths
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_PATH = REPORTS_DIR / "environment_report.md"
HTML_PATH = REPORTS_DIR / "environment_report.html"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return proc.stdout.strip()


def summarise_gpu(gpu_report: Dict[str, Any]) -> tuple[str, str, str, List[str]]:
    warnings = list(gpu_report.get("warnings", []))
    gpu_details = gpu_report.get("gpu", {}).get("details", [])
    if gpu_details:
        name = gpu_details[0].get("name", "unknown")
        driver = gpu_details[0].get("driver_version", "unknown")
        cuda = gpu_details[0].get("cuda_version", gpu_report.get("detected_cuda_version", "unknown"))
    else:
        name = "N/A"
        driver = "N/A"
        cuda = gpu_report.get("detected_cuda_version") or "N/A"

    torch_info = gpu_report.get("torch", {})
    cudnn = torch_info.get("cudnn_version") or "N/A"
    nccl = torch_info.get("nccl_version") or "N/A"
    if torch_info.get("warnings"):
        warnings.extend(torch_info["warnings"])

    return name, driver, cuda, warnings + gpu_report.get("warnings", [])


def summarise_dependencies(lockfile: Path, pip_tree: Path) -> tuple[str, str]:
    lock_preview = "\n".join(read_text(lockfile).splitlines()[:20])
    pip_tree_preview = "\n".join(read_text(pip_tree).splitlines()[:40])
    return lock_preview, pip_tree_preview


def summarise_inference(test_summary_path: Path) -> tuple[str, List[str], Dict[str, Any]]:
    summary = read_json(test_summary_path)
    status = summary.get("status", "unknown")
    warnings = summary.get("warnings", [])
    return status, warnings, summary.get("results", {})


def to_html(markdown_text: str) -> str:
    try:
        import markdown  # type: ignore

        return markdown.markdown(markdown_text)
    except ModuleNotFoundError:
        # Simple fallback: wrap markdown in <pre>
        from html import escape

        return f"<html><body><pre>{escape(markdown_text)}</pre></body></html>"


def main() -> int:
    gpu_report = read_json(REPORTS_DIR / "system_gpu.json")

    torch_info = gpu_report.get("torch", {})
    torch_installed = torch_info.get("torch_installed")
    if not torch_installed:
        try:
            import torch  # type: ignore

            cudnn_ver = getattr(torch.backends.cudnn, "version", lambda: None)()
            if callable(cudnn_ver):
                cudnn_ver = cudnn_ver()
            nccl_ver = None
            if hasattr(torch.cuda, "nccl") and torch.cuda.is_available():
                nccl_ver = torch.cuda.nccl.version()

            gpu_report["torch"] = {
                "torch_installed": torch.__version__,
                "cudnn_version": str(cudnn_ver) if cudnn_ver else None,
                "nccl_version": str(nccl_ver) if nccl_ver else None,
            }

            warnings_list = gpu_report.get("warnings", [])
            cleaned = [w for w in warnings_list if "torch not installed" not in w]
            if len(cleaned) != len(warnings_list):
                gpu_report["warnings"] = cleaned
        except ModuleNotFoundError:
            pass
    test_status, test_warnings, test_results = summarise_inference(REPORTS_DIR / "test_summary.json")
    lock_preview, pip_tree_preview = summarise_dependencies(ROOT / "requirements.lock", REPORTS_DIR / "pip_tree.txt")
    gpu_name, driver_version, cuda_version, gpu_warnings = summarise_gpu(gpu_report)

    python_version = platform.python_version()
    pip_version = run(["pip", "--version"])
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

    model_name, _, _ = get_model_identifiers()
    model_paths = get_model_paths()
    weights_dir = model_paths["weights"]
    tokenizer_dir = model_paths["tokenizer"]
    metadata_dir = model_paths["metadata"]

    warnings = list(dict.fromkeys(gpu_warnings + test_warnings))

    markdown_lines = [
        "# Environment Verification Report",
        "",
        f"**Model:** {model_name}",
        f"**Last Verified:** {timestamp}",
        f"**GPU:** {gpu_name}",
        f"**Driver:** {driver_version}",
        f"**CUDA:** {cuda_version}",
        f"**Python:** {python_version}",
        f"**pip:** {pip_version}",
        "",
        "**Assets:**",
        f"- Weights: {weights_dir}",
        f"- Tokenizer: {tokenizer_dir}",
        f"- Metadata: {metadata_dir}",
        "",
    ]

    torch_info = gpu_report.get("torch", {})
    torch_version = torch_info.get("torch_installed") or "N/A"
    cudnn = torch_info.get("cudnn_version") or "N/A"
    nccl = torch_info.get("nccl_version") or "N/A"

    markdown_lines.extend(
        [
            f"**torch:** {torch_version}",
            f"**cuDNN:** {cudnn}",
            f"**NCCL:** {nccl}",
            "",
            f"**Sanity Test:** {test_status}",
            f"**Sanity Metrics:** {json.dumps(test_results) if test_results else 'N/A'}",
            "",
        ]
    )

    asset_problems: List[str] = []
    pairs = [
        ("weights", weights_dir),
        ("tokenizer", tokenizer_dir),
        ("metadata", metadata_dir),
    ]
    for label, path in pairs:
        if not path.exists():
            asset_problems.append(f"{label}: missing directory {path}")
        else:
            if label == "weights":
                has_shards = any(path.glob("model-*.safetensors"))
                if not has_shards:
                    asset_problems.append(f"{label}: no model-*.safetensors files in {path}")
            if label == "tokenizer":
                tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
                missing = [name for name in tokenizer_files if not (path / name).exists()]
                if missing:
                    asset_problems.append(f"{label}: missing {', '.join(missing)} in {path}")
            if label == "metadata":
                metadata_files = ["config.json", "generation_config.json", "model.safetensors.index.json"]
                missing = [name for name in metadata_files if not (path / name).exists()]
                if missing:
                    asset_problems.append(f"{label}: missing {', '.join(missing)} in {path}")

    if asset_problems:
        warnings.extend(asset_problems)
        warnings.append("Run `python scripts/download_weights.py --model-name olmo2` to restage assets.")

    if warnings:
        markdown_lines.append("## Warnings")
        markdown_lines.extend([f"- {warning}" for warning in warnings])
        markdown_lines.append("")

    if test_results:
        markdown_lines.extend(
            [
                "## Sanity Test Details",
                "```json",
                json.dumps(test_results, indent=2),
                "```",
                "",
            ]
        )

    test_markdown = REPORTS_DIR / "test_summary.md"
    if test_markdown.exists():
        markdown_lines.extend(
            [
                "## Test Summary",
                test_markdown.read_text().strip(),
                "",
            ]
        )

    markdown_lines.extend(
        [
            "## Locked Dependencies (first 20)",
            "```",
            lock_preview or "requirements.lock not generated.",
            "```",
            "",
            "## Dependency Tree (first 40)",
            "```",
            pip_tree_preview or "pipdeptree output not available.",
            "```",
        ]
    )

    markdown_text = "\n".join(markdown_lines).strip() + "\n"
    MARKDOWN_PATH.write_text(markdown_text)
    HTML_PATH.write_text(to_html(markdown_text))

    print(f"[run_env_report] Markdown report -> {MARKDOWN_PATH.relative_to(ROOT)}")
    print(f"[run_env_report] HTML report -> {HTML_PATH.relative_to(ROOT)}")
    if warnings:
        print("[run_env_report] Warnings included in report:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("[run_env_report] No warnings detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
