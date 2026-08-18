from pathlib import Path
import subprocess
import modal

app = modal.App("cs336-a2-benchmark")


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
    modal.Image.debian_slim(python_version="3.11")
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

@app.function(image=image, gpu="T4", timeout=20 * 60)
def smoke_test():
    subprocess.run(
        [
            "uv", "run", "cs336_systems/benchmark.py",
            "--context-length", "128",
            "--d-model", "128",
            "--num-layers", "2",
            "--num-heads", "4",
            "--batch-size", "2",
            "--d-ff", "512",
            "--warmup-steps", "2",
            "--steps", "3",
            "--backward",
        ],
        cwd="/root/assignment2",
        check=True,
    )

@app.local_entrypoint()
def main():
    smoke_test.remote()