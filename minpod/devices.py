from __future__ import annotations

import os

import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        torch.backends.mps.is_available()
        and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    ):
        return "mps"
    return "cpu"


def device_label(device: str) -> str:
    if device == "mps":
        return "apple silicon (mps)"
    return device
