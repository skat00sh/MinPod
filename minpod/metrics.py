from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass
class PipelineMetrics:
    device: str
    load_s: float
    ttft_s: float
    gen_s: float
    audio_s: float
    cpu_pct: float
    ram_gb: float
    vram_gb: float | None = None

    @property
    def rtf(self) -> float:
        if self.audio_s <= 0:
            return 0.0
        return self.gen_s / self.audio_s

    def format(self, output_path: str) -> str:
        rows = [
            ("output", output_path),
            ("device", self.device),
            ("load", f"{self.load_s:.1f}s"),
            ("ttft", f"{self.ttft_s:.2f}s"),
            ("gen", f"{self.gen_s:.1f}s  rtf {self.rtf:.2f}"),
            ("audio", f"{self.audio_s:.1f}s"),
            ("cpu", f"{self.cpu_pct:.0f}%"),
            ("ram", f"{self.ram_gb:.1f} GB"),
        ]
        if self.vram_gb is not None:
            rows.append(("vram", f"{self.vram_gb:.1f} GB"))
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"{name:<{width}}  {value}" for name, value in rows)


class ResourceMonitor:
    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._ram_peak = 0
        self._use_cuda = False

    def start(self) -> None:
        self._proc.cpu_percent(interval=None)
        self._ram_peak = self._proc.memory_info().rss
        try:
            import torch

            self._use_cuda = torch.cuda.is_available()
            if self._use_cuda:
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            self._use_cuda = False

    def stop(self) -> tuple[float, float, float | None]:
        cpu_pct = self._proc.cpu_percent(interval=None)
        ram_now = self._proc.memory_info().rss
        ram_gb = max(self._ram_peak, ram_now) / (1024**3)
        vram_gb = None
        if self._use_cuda:
            import torch

            vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        return cpu_pct, ram_gb, vram_gb
