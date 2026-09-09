import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.optimizer_state_sharding import OptimizerStateSharding
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import DDP


WORLD_SIZE = 2
BATCH_SIZE = 4          # global batch size
SEQ_LEN = 512
VOCAB_SIZE = 10_000

# XL configuration from Section 2.1.2
D_MODEL = 2560
NUM_LAYERS = 32
D_FF = 10240
NUM_HEADS = 32


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def bytes_to_gib(n):
    return n / (1024 ** 3)


def benchmark(rank, world_size, sharded):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    local_batch_size = BATCH_SIZE // world_size

    # ------------------------------------------------------------
    # 1. Model initialization
    # ------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=SEQ_LEN,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        num_heads=NUM_HEADS,
    ).to(device)

    model = DDP(model)

    if sharded:
        optimizer = OptimizerStateSharding(
            model.parameters(),
            AdamW,
        )
    else:
        optimizer = AdamW(model.parameters())

    torch.cuda.synchronize(device)

    peak_after_model_init = torch.cuda.max_memory_allocated(device)

    # ------------------------------------------------------------
    # Create inputs
    # ------------------------------------------------------------
    torch.manual_seed(42 + rank)

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (local_batch_size, SEQ_LEN),
        dtype=torch.long,
        device=device,
    )

    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (local_batch_size, SEQ_LEN),
        dtype=torch.long,
        device=device,
    )

    optimizer.zero_grad()

    # ------------------------------------------------------------
    # 2. Forward + backward
    #    Measure peak directly before optimizer.step()
    # ------------------------------------------------------------
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    logits = model(input_ids)
    loss = cross_entropy(logits, targets)

    loss.backward()
    model.finish_gradient_synchronization()

    torch.cuda.synchronize(device)

    peak_before_optimizer_step = torch.cuda.max_memory_allocated(device)

    # ------------------------------------------------------------
    # 3. Optimizer step
    #
    # AdamW initializes optimizer state on the first step, so this
    # is where sharding should have its largest effect.
    # ------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)

    optimizer.step()

    torch.cuda.synchronize(device)

    peak_after_optimizer_step = torch.cuda.max_memory_allocated(device)

    # ------------------------------------------------------------
    # Gather results onto rank 0
    # ------------------------------------------------------------
    result = torch.tensor(
        [
            peak_after_model_init,
            peak_before_optimizer_step,
            peak_after_optimizer_step,
        ],
        dtype=torch.float64,
        device=device,
    )

    gathered = (
        [torch.zeros_like(result) for _ in range(world_size)]
        if rank == 0
        else None
    )

    dist.gather(result, gather_list=gathered, dst=0)

    if rank == 0:
        mode = "sharded" if sharded else "unsharded"

        print(f"\n=== {mode.upper()} ===")

        for gpu_rank, values in enumerate(gathered):
            print(
                f"GPU {gpu_rank}:\n"
                f"  after model initialization: "
                f"{bytes_to_gib(values[0].item()):.3f} GiB\n"
                f"  before optimizer step:      "
                f"{bytes_to_gib(values[1].item()):.3f} GiB\n"
                f"  after optimizer step:       "
                f"{bytes_to_gib(values[2].item()):.3f} GiB"
            )

    dist.destroy_process_group()


def run(sharded):
    mp.spawn(
        benchmark,
        args=(WORLD_SIZE, sharded),
        nprocs=WORLD_SIZE,
        join=True,
    )


def main():
    assert BATCH_SIZE % WORLD_SIZE == 0

    print("Running WITHOUT optimizer state sharding...")
    run(sharded=False)

    print("\nRunning WITH optimizer state sharding...")
    run(sharded=True)


if __name__ == "__main__":
    main()