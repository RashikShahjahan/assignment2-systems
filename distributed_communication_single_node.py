import os
import timeit
import statistics
import matplotlib.pyplot as plt

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DATA_SIZES = {
    "1 MB": 1_000_000 // 4,
    "10 MB": 10_000_000 // 4,
    "100 MB": 100_000_000 // 4,
    "1 GB": 1_000_000_000 // 4,
}


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device=rank
    )


def benchmark(rank, world_size, shared_results):
    setup(rank, world_size)

    for label, num_elements in DATA_SIZES.items():
        data = torch.rand(
            num_elements,
            dtype=torch.float32,
        )

        # Warmup
        for _ in range(5):
            dist.all_reduce(data, async_op=False)

        local_measurements = []

        for _ in range(10):
            torch.cuda.synchronize()

            start_time = timeit.default_timer()

            dist.all_reduce(data, async_op=False)
            torch.cuda.synchronize()

            elapsed = timeit.default_timer() - start_time

            # Convert seconds -> milliseconds
            local_measurements.append(elapsed * 1000)

        gathered_measurements = [None] * world_size

        dist.all_gather_object(
            gathered_measurements,
            local_measurements,
        )

        if rank == 0:
            all_measurements = [
                measurement
                for rank_measurements in gathered_measurements
                for measurement in rank_measurements
            ]

            shared_results.append({
                "world_size": world_size,
                "size": label,
                "size_mb": num_elements * 4 / 1_000_000,
                "mean_ms": statistics.mean(all_measurements),
                "std_ms": statistics.stdev(all_measurements),
            })

    dist.destroy_process_group()


if __name__ == "__main__":
    manager = mp.Manager()
    results = manager.list()

    for world_size in [2, 4, 6]:
        mp.spawn(
            fn=benchmark,
            args=(world_size, results),
            nprocs=world_size,
            join=True,
        )

    df = pd.DataFrame(list(results))

    df = df.sort_values(["world_size", "size_mb"])

    print(df)


    for world_size, group in df.groupby("world_size"):
        plt.errorbar(
            group["size_mb"],
            group["mean_ms"],
            yerr=group["std_ms"],
            marker="o",
            capsize=3,
            label=f"{world_size} processes",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Tensor size (MB)")
    plt.ylabel("All-reduce runtime (ms)")
    plt.legend()
    plt.tight_layout()
    plt.show()