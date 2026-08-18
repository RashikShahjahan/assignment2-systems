from pathlib import Path
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
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("ca-certificates", "gnupg")
    .run_commands(
        "echo 'deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' "
        "> /etc/apt/sources.list.d/nvidia-devtools.list",
        "apt-key adv --fetch-keys "
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub",
        "apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-cli",
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
            "--output=/profiles/benchmark",
            "--force-overwrite=true",
            "--", "python", "cs336_systems/benchmark.py",
            "--context-length", "512",
            "--d-model", "768",
            "--num-layers", "12",
            "--num-heads", "12",
            "--batch-size", "4",
            "--d-ff", "3072",
            "--warmup-steps", "5",
            "--steps", "10",
            "--backward",
            "--optimizer"
        ],
        cwd="/root/assignment2",
        check=True,
    )
    profiles.commit()

@app.local_entrypoint()
def main():
    smoke_test.remote()
