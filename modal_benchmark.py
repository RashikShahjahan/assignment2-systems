from pathlib import Path
import shutil
import subprocess
import modal

app = modal.App("cs336-a2-benchmark")
profiles = modal.Volume.from_name("cs336-nsys-profiles", create_if_missing=True)


ROOT = Path(__file__).resolve().parent

EXCLUDED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

def ignore_project_file(path: Path) -> bool:
    return (
        any(part in EXCLUDED_DIRS for part in path.parts)
        or path.suffix in {".pyc", ".pyo"}
        or path.name == ".DS_Store"
    )

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.14")
    .entrypoint([])
    .apt_install("binutils", "ca-certificates", "gnupg")
    .run_commands(
        "echo 'deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' "
        "> /etc/apt/sources.list.d/nvidia-devtools.list",
        "apt-key adv --fetch-keys "
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub",
        "apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-cli-2026.4.1",
    )
    .add_local_file(
        ROOT / "nsys-config.ini",
        remote_path="/root/.config/NVIDIA Corporation/nsys-config.ini",
        copy=True,
    )
    .uv_pip_install("uv")
    .add_local_dir(
        ROOT,
        remote_path="/root/assignment2",
        copy=True,
        ignore=ignore_project_file,
    )
    .run_commands(
        # This check occurs before uv creates the Linux environment.
        "test ! -e /root/assignment2/.venv",
        "cd /root/assignment2 && uv sync --frozen",
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=20 * 60,
    volumes={"/profiles": profiles},
)
def smoke_test():
    subprocess.run(
        [
            "uv", "run", "nsys", "profile",
            "--trace=cuda-sw,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-um-cpu-page-faults=false",
            "--cuda-um-gpu-page-faults=false",
            "--cuda-flush-interval=100",
            "--capture-range=nvtx",
            "--capture-range-end=stop",
            "--nvtx-capture=profile",
            "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
            "--output=/profiles/benchmark",
            "--force-overwrite=true",
            "--stats=true",
            "--", "python", "cs336_systems/benchmark.py",
            "--context-length", "512",
            "--d-model", "768",
            "--num-layers", "12",
            "--num-heads", "12",
            "--batch-size", "4",
            "--d-ff", "3072",
            "--warmup-steps", "5",
            "--steps", "1",
            "--backward",
            "--optimizer",
             "--bf16",
        ],
        cwd="/root/assignment2",
        check=True,
    )
    shutil.copy2(
        "/root/assignment2/memory_snapshot.pickle",
        "/profiles/memory_snapshot.pickle",
    )
    profiles.commit()

@app.local_entrypoint()
def main():
    smoke_test.remote()
