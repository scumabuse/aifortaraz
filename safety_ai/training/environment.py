"""Training environment detection and ``reports/environment.md`` (docs/TRAINING_SPEC.md 6.1).

Mode rule of the specification: **GPU** when CUDA is available with at least 6 GB of VRAM,
otherwise **CPU** (an Apple MPS device counts as CPU mode, with ``device="mps"`` tried first and
the CPU as the fallback). The VRAM threshold compares the card's total memory in GiB rounded to
one decimal, so a "6 GB" card reporting 6144 MiB qualifies (DECISIONS.md T-004).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

import psutil
import torch
import ultralytics

__all__ = [
    "GPU_MIN_VRAM_GB",
    "Environment",
    "GpuInfo",
    "choose_mode",
    "cpu_model",
    "detect_environment",
    "nvidia_smi_info",
    "render_environment",
]

GPU_MIN_VRAM_GB: Final[float] = 6.0
Mode = Literal["gpu", "cpu"]
_GIB: Final[float] = float(1 << 30)


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """One CUDA device as torch and nvidia-smi see it."""

    index: int
    name: str
    vram_total_mib: int
    vram_used_mib: int | None
    driver: str | None
    compute_capability: str


@dataclass(frozen=True, slots=True)
class Environment:
    """Everything spec 6.1 asks to record, plus the chosen mode and why."""

    detected_at: str
    os: str
    python: str
    torch: str
    torch_cuda: str | None
    cudnn: int | None
    ultralytics: str
    cpu: str
    cpu_logical: int
    cpu_physical: int | None
    ram_total_gib: float
    ram_available_gib: float
    cuda_available: bool
    gpus: tuple[GpuInfo, ...]
    mps_available: bool
    mode: Mode
    device: str
    reason: str

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        data = asdict(self)
        data["gpus"] = [asdict(g) for g in self.gpus]
        return data


def nvidia_smi_info(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[int, tuple[int | None, str | None]]:
    """``{gpu index: (used MiB, driver version)}`` from nvidia-smi; empty when unavailable."""
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        done = run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    info: dict[int, tuple[int | None, str | None]] = {}
    for line in done.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            used = int(parts[1]) if parts[1].isdigit() else None
            info[int(parts[0])] = (used, parts[2] or None)
    return info


def cpu_model() -> str:
    """Marketing name of the CPU (Windows registry / /proc/cpuinfo / sysctl), else platform."""
    if sys.platform == "win32":
        try:
            import winreg  # noqa: PLC0415 - Windows-only module

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            return platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "darwin" and shutil.which("sysctl"):
        done = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.stdout.strip():
            return done.stdout.strip()
    return platform.processor() or platform.machine()


def choose_mode(gpus: tuple[GpuInfo, ...], mps_available: bool) -> tuple[Mode, str, str]:
    """Apply the spec 6.1 rule; return ``(mode, device, reason)``."""
    for gpu in gpus:
        total_gib = round(gpu.vram_total_mib / 1024.0, 1)
        if total_gib >= GPU_MIN_VRAM_GB:
            return (
                "gpu",
                f"cuda:{gpu.index}",
                f"CUDA device {gpu.index} ({gpu.name}) has {total_gib:.1f} GiB VRAM >= "
                f"{GPU_MIN_VRAM_GB:.0f} GB",
            )
    if gpus:
        best = max(gpus, key=lambda g: g.vram_total_mib)
        reason = f"CUDA VRAM {best.vram_total_mib / 1024:.1f} GiB < {GPU_MIN_VRAM_GB:.0f} GB"
    else:
        reason = "no CUDA device"
    if mps_available:
        return "cpu", "mps", f"{reason}; Apple MPS counts as CPU mode (mps tried first)"
    return "cpu", "cpu", reason


def detect_environment(*, clock: Callable[[], datetime] | None = None) -> Environment:
    """Detect the hardware and software of this machine (spec 6.1)."""
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    smi = nvidia_smi_info()
    gpus: list[GpuInfo] = []
    cuda = torch.cuda.is_available()
    if cuda:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            used, driver = smi.get(index, (None, None))
            gpus.append(
                GpuInfo(
                    index=index,
                    name=str(props.name),
                    vram_total_mib=int(props.total_memory // (1 << 20)),
                    vram_used_mib=used,
                    driver=driver,
                    compute_capability=f"{props.major}.{props.minor}",
                )
            )
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    mode, device, reason = choose_mode(tuple(gpus), mps)
    memory = psutil.virtual_memory()
    cudnn = torch.backends.cudnn.version() if cuda else None
    return Environment(
        detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        os=platform.platform(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_cuda=torch.version.cuda,
        cudnn=int(cudnn) if cudnn else None,
        ultralytics=str(ultralytics.__version__),
        cpu=cpu_model(),
        cpu_logical=os.cpu_count() or 1,
        cpu_physical=psutil.cpu_count(logical=False),
        ram_total_gib=round(memory.total / _GIB, 1),
        ram_available_gib=round(memory.available / _GIB, 1),
        cuda_available=cuda,
        gpus=tuple(gpus),
        mps_available=mps,
        mode=mode,
        device=device,
        reason=reason,
    )


def render_environment(env: Environment, *, command: str) -> str:
    """Markdown of ``reports/environment.md``."""
    lines = [
        "# Training environment (docs/TRAINING_SPEC.md 6.1)",
        "",
        f"Detected {env.detected_at} with `{command}`.",
        "",
        "| item | value |",
        "|---|---|",
        f"| OS | {env.os} |",
        f"| Python | {env.python} |",
        f"| torch | {env.torch} (CUDA build {env.torch_cuda or 'none'}, "
        f"cuDNN {env.cudnn or '-'}) |",
        f"| ultralytics | {env.ultralytics} |",
        f"| CPU | {env.cpu}, {env.cpu_physical or '?'} cores / {env.cpu_logical} threads |",
        f"| RAM | {env.ram_total_gib:.1f} GiB total, {env.ram_available_gib:.1f} GiB available |",
        f"| CUDA available | {'yes' if env.cuda_available else 'no'} |",
    ]
    for gpu in env.gpus:
        used = f", {gpu.vram_used_mib} MiB in use" if gpu.vram_used_mib is not None else ""
        lines.append(
            f"| GPU {gpu.index} | {gpu.name}, {gpu.vram_total_mib} MiB VRAM{used}, driver "
            f"{gpu.driver or '?'}, compute capability {gpu.compute_capability} |"
        )
    mps = "yes" if env.mps_available else "n/a (not an Apple Silicon machine)"
    lines += [
        f"| MPS | {mps} |",
        f"| **mode** | **{env.mode.upper()}**, device `{env.device}`: {env.reason} |",
        "",
        "Rule (spec 6.1): GPU mode when CUDA has >= 6 GB VRAM, otherwise CPU mode (MPS counts "
        'as CPU mode with `device="mps"` tried first).',
        "",
    ]
    return "\n".join(lines)


def write_environment_report(out: Path, *, command: str) -> Environment:
    """Detect the environment and write the markdown report plus a JSON twin next to it."""
    env = detect_environment()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_environment(env, command=command), encoding="utf-8", newline="\n")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(env.to_json(), indent=2) + "\n", encoding="utf-8", newline="\n")
    return env


def main(argv: list[str] | None = None) -> int:
    """``python -m safety_ai.training.environment [--out reports/environment.md]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path("reports/environment.md")
    if args[:1] == ["--out"] and len(args) == 2:
        out = Path(args[1])
    elif args:
        print("usage: python -m safety_ai.training.environment [--out PATH]")
        return 2
    command = "python -m safety_ai.training.environment" + (f" --out {args[1]}" if args else "")
    env = write_environment_report(out, command=command)
    print(f"mode {env.mode} ({env.device}): {env.reason}; report {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
