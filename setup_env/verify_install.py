#!/usr/bin/env python3
"""Verify torch installation against detected CUDA version."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "verify_install.json"
AUTODETECTED_ENV = ROOT / ".env.autodetected"


def load_autodetected_cuda() -> Optional[str]:
    if "CUDA_VERSION" in os.environ and os.environ["CUDA_VERSION"]:
        return os.environ["CUDA_VERSION"]
    if AUTODETECTED_ENV.exists():
        for line in AUTODETECTED_ENV.read_text().splitlines():
            if line.startswith("CUDA_VERSION="):
                value = line.split("=", 1)[1].strip()
                return value or None
    return None


def parse_cuda_version(cuda_version: str | None) -> Optional[Tuple[int, int]]:
    if not cuda_version:
        return None
    parts = cuda_version.strip().split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None
    return major, minor


def select_wheel(major_minor: Optional[Tuple[int, int]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (tag, torch_cuda_version, note)."""
    if major_minor is None:
        return None, None, None

    mapping = {
        (11, 8): ("cu118", "11.8", None),
        (12, 1): ("cu121", "12.1", None),
        (12, 4): ("cu124", "12.4", None),
        (12, 8): ("cu128", "12.8", None),
    }

    if major_minor in mapping:
        tag, torch_cuda, note = mapping[major_minor]
        return tag, torch_cuda, note

    major, minor = major_minor
    if major >= 13:
        note = (
            "CUDA %d.%d detected; PyTorch does not yet publish a matching wheel. "
            "Falling back to the CUDA 12.8 build which includes SM_120 support."
        ) % (major, minor)
        return "cu128", "12.8", note

    return None, None, "No matching PyTorch wheel available for CUDA %d.%d." % (major, minor)


def inspect_torch() -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "version": None,
        "cuda": None,
    }
    try:
        import torch  # type: ignore

        info["version"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None)
    except ModuleNotFoundError:
        pass
    return info


def build_report() -> Dict[str, Any]:
    detected_cuda = load_autodetected_cuda()
    torch_info = inspect_torch()
    torch_cuda = torch_info.get("cuda")
    major_minor = parse_cuda_version(detected_cuda)
    target_tag, target_cuda_version, note = select_wheel(major_minor)
    needs_reinstall = False
    explanation = ""
    notes = []
    if note:
        notes.append(note)

    if torch_info["version"] is None:
        explanation = "PyTorch not installed."
        needs_reinstall = True
    elif detected_cuda and torch_cuda and torch_cuda != detected_cuda:
        if target_cuda_version and torch_cuda == target_cuda_version:
            explanation = (
                f"Torch CUDA {torch_cuda} differs from detected CUDA {detected_cuda}, "
                "but is the closest available build."
            )
            needs_reinstall = False
        elif target_tag is None:
            explanation = (
                f"Detected CUDA {detected_cuda} but no matching PyTorch wheel is available. "
                "Continuing with the installed build."
            )
            needs_reinstall = False
        else:
            explanation = (
                f"Installed torch CUDA {torch_cuda} does not match detected CUDA {detected_cuda}."
            )
            needs_reinstall = True
    elif detected_cuda and torch_cuda is None:
        if target_tag is None:
            explanation = (
                "Detected CUDA version but installed torch build has no CUDA support. "
                "No compatible wheel detected; install manually if GPU support is required."
            )
            needs_reinstall = False
        else:
            explanation = (
                "Detected CUDA version but installed torch build has no CUDA support."
            )
            needs_reinstall = True
    elif detected_cuda and torch_cuda and target_cuda_version and torch_cuda != target_cuda_version:
        # Current torch doesn't match closest target (e.g. CPU-only build).
        explanation = (
            f"Installed torch CUDA {torch_cuda} differs from expected CUDA {target_cuda_version}."
        )
        needs_reinstall = True

    suggested_url = None
    if target_tag:
        suggested_url = f"https://download.pytorch.org/whl/{target_tag}"

    if needs_reinstall and suggested_url is None and torch_info["version"] is not None:
        needs_reinstall = False
        if explanation:
            notes.append(explanation)
        explanation = ""

    report = {
        "detected_cuda": detected_cuda,
        "torch_version": torch_info["version"],
        "torch_cuda": torch_cuda,
        "needs_reinstall": needs_reinstall,
        "suggested_index_url": suggested_url,
        "explanation": explanation,
        "notes": notes,
    }
    return report


def emit(report: Dict[str, Any], json_only: bool = False) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    if json_only:
        print(json.dumps(report))
        return

    detected_cuda = report.get("detected_cuda") or "unknown"
    torch_version = report.get("torch_version")
    torch_cuda = report.get("torch_cuda") or "unknown"

    if torch_version:
        print(f"✅ torch {torch_version} reports CUDA {torch_cuda}")
    else:
        print("❌ torch not installed.")

    print(f"ℹ️  Detected CUDA version: {detected_cuda}")

    if report.get("needs_reinstall"):
        print("❌ Torch/CUDA mismatch detected.")
        suggestion = report.get("suggested_index_url")
        if suggestion:
            print(f"   → Reinstall with: pip install --upgrade torch torchvision torchaudio --index-url {suggestion}")
        explanation = report.get("explanation")
        if explanation:
            print(f"   → {explanation}")
    else:
        print("✅ Torch build matches detected CUDA toolchain.")

    for note in report.get("notes", []):
        print(f"ℹ️  {note}")

    print(f"Report saved to {REPORT_PATH.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify torch ↔ CUDA compatibility.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON only.")
    args = parser.parse_args()

    report = build_report()
    emit(report, json_only=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
