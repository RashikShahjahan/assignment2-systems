from __future__ import annotations

import csv
import gc
import itertools
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENGTHS = [256, 1024, 4096, 8192, 16384]

NUM_RUNS = 100
WARMUP_RUNS = 5
DTYPE = torch.float32
DEVICE = torch.device("cuda")


def synchronize() -> None:
    torch.cuda.synchronize(DEVICE)


def clear_gradients(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def benchmark_configuration(seq_len: int, d_model: int) -> dict:
    shape = (BATCH_SIZE, seq_len, d_model)

    q = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Warm up both forward and backward kernels.
    for _ in range(WARMUP_RUNS):
        output = scaled_dot_product_attention(q, k, v)
        synchronize()

        loss = output.sum()
        loss.backward()
        synchronize()

        clear_gradients(q, k, v)
        del output, loss

    # Time forward passes.
    forward_seconds = 0.0

    for _ in range(NUM_RUNS):
        start = timeit.default_timer()

        output = scaled_dot_product_attention(q, k, v)
        synchronize()

        forward_seconds += timeit.default_timer() - start
        del output

    # Measure live memory after forward and immediately before backward.
    gc.collect()
    torch.cuda.empty_cache()

    baseline_bytes = torch.cuda.memory_allocated(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)

    output = scaled_dot_product_attention(q, k, v)
    synchronize()
    loss = output.sum()
    synchronize()

    memory_before_backward_bytes = torch.cuda.memory_allocated(DEVICE)
    peak_forward_bytes = torch.cuda.max_memory_allocated(DEVICE)

    memory_before_backward_mib = memory_before_backward_bytes / 1024**2
    forward_memory_delta_mib = (
        memory_before_backward_bytes - baseline_bytes
    ) / 1024**2
    forward_peak_delta_mib = (
        peak_forward_bytes - baseline_bytes
    ) / 1024**2

    del output, loss

    # Time backward passes. Each iteration needs a fresh graph.
    backward_seconds = 0.0

    for _ in range(NUM_RUNS):
        clear_gradients(q, k, v)

        output = scaled_dot_product_attention(q, k, v)
        synchronize()

        loss = output.sum()
        synchronize()

        start = timeit.default_timer()

        loss.backward()
        synchronize()

        backward_seconds += timeit.default_timer() - start
        del output, loss

    return {
        "batch_size": BATCH_SIZE,
        "sequence_length": seq_len,
        "d_model": d_model,
        "forward_mean_ms": forward_seconds / NUM_RUNS * 1000,
        "backward_mean_ms": backward_seconds / NUM_RUNS * 1000,
        "memory_before_backward_mib": memory_before_backward_mib,
        "forward_memory_delta_mib": forward_memory_delta_mib,
        "forward_peak_delta_mib": forward_peak_delta_mib,
        "status": "ok",
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    results = []

    for d_model, seq_len in itertools.product(HEAD_DIMS, SEQ_LENGTHS):
        print(f"Benchmarking d_model={d_model}, sequence_length={seq_len}")

        try:
            result = benchmark_configuration(seq_len, d_model)
            print(
                f"  forward={result['forward_mean_ms']:.3f} ms, "
                f"backward={result['backward_mean_ms']:.3f} ms, "
                f"memory={result['memory_before_backward_mib']:.1f} MiB"
            )
        except torch.OutOfMemoryError:
            result = {
                "batch_size": BATCH_SIZE,
                "sequence_length": seq_len,
                "d_model": d_model,
                "forward_mean_ms": "",
                "backward_mean_ms": "",
                "memory_before_backward_mib": "",
                "forward_memory_delta_mib": "",
                "forward_peak_delta_mib": "",
                "status": "oom",
            }
            print("  CUDA out of memory")

        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    with open("attention_benchmark.csv", "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()