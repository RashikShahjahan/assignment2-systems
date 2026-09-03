import os
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import DDP


WORLD_SIZE = 2

# Global batch size.
# Each GPU receives BATCH_SIZE / WORLD_SIZE examples.
BATCH_SIZE = 4
SEQ_LEN = 512
VOCAB_SIZE = 10_000


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def benchmark(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # XL model from Section 2.1.2
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=SEQ_LEN,
        d_model=2560,
        num_layers=32,
        d_ff=10240,
        num_heads=32,
    ).to(device)

    model = DDP(model)

    optimizer = AdamW(model.parameters())

    torch.manual_seed(42 + rank)

    local_batch_size = BATCH_SIZE // world_size

    input_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(local_batch_size, SEQ_LEN),
        dtype=torch.long,
        device=device,
    )

    targets = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(local_batch_size, SEQ_LEN),
        dtype=torch.long,
        device=device,
    )

    # -------------------------
    # Warmup
    # -------------------------
    for _ in range(5):
        optimizer.zero_grad()

        logits = model(input_ids)
        loss = cross_entropy(logits, targets)

        loss.backward()
        model.finish_gradient_synchronization()

        optimizer.step()

    torch.cuda.synchronize()

    # Make sure ranks start benchmark together
    dist.barrier()

    step_measurements = []
    comm_measurements = []

    # -------------------------
    # Benchmark
    # -------------------------
    for _ in range(10):
        torch.cuda.synchronize()
        start = timeit.default_timer()

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = cross_entropy(logits, targets)

        loss.backward()

        # Measure gradient communication separately
        torch.cuda.synchronize()
        comm_start = timeit.default_timer()

        model.finish_gradient_synchronization()

        torch.cuda.synchronize()
        comm_elapsed = timeit.default_timer() - comm_start

        optimizer.step()

        torch.cuda.synchronize()
        step_elapsed = timeit.default_timer() - start

        step_measurements.append(step_elapsed)
        comm_measurements.append(comm_elapsed)

    mean_step = statistics.mean(step_measurements)
    mean_comm = statistics.mean(comm_measurements)

    if rank == 0:
        print(f"Mean training step: {mean_step * 1000:.3f} ms")
        print(f"Mean communication: {mean_comm * 1000:.3f} ms")
        print(f"Communication proportion: {mean_comm / mean_step * 100:.2f}%")

        print(
            f"Step std: {statistics.stdev(step_measurements) * 1000:.3f} ms"
        )
        print(
            f"Communication std: "
            f"{statistics.stdev(comm_measurements) * 1000:.3f} ms"
        )

    dist.destroy_process_group()


def main():
    mp.spawn(
        benchmark,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()