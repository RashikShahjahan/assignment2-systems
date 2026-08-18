import statistics

import argparse
import torch
import timeit
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


def main():
    parser = argparse.ArgumentParser(description='benchmark language models')

    parser.add_argument('--context-length', type=int)
    parser.add_argument('--d-model', type=int)
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--num-heads', type=int)
    parser.add_argument('--batch-size', type=int)

    parser.add_argument('--d-ff', type=int)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument(
    "--backward",
    action="store_true",
    help="Benchmark backward pass",)    
    parser.add_argument(
    "--optimizer",
    action="store_true",
    help="Benchmark optimizer step",)

    args = parser.parse_args()

    if args.optimizer and not args.backward:
        parser.error("--optimizer requires --backward")

    device = torch.device("cuda")
    model = BasicsTransformerLM(vocab_size=10000, context_length= args.context_length, d_model=args.d_model,num_layers=args.num_layers, d_ff=args.d_ff, num_heads = args.num_heads).to(device)

    optimizer = AdamW(model.parameters())

    torch.manual_seed(42)
    input_ids = torch.randint(
        0,
        10000,
        ( args.batch_size,  args.context_length),
        dtype=torch.long,

        device = device
    )

    targets = torch.randint(
    low=0,
    high=10_000,
    size=(args.batch_size, args.context_length),
    dtype=torch.long,
    device=device,
    )

    measurements = []

    for t in range(args.warmup_steps+args.steps):
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        if args.backward:
            optimizer.zero_grad()
        logits = model(input_ids)
        if args.backward:
            loss = cross_entropy(logits, targets)
            loss.backward()
            if args.optimizer:
                optimizer.step()
        torch.cuda.synchronize()
        elapsed = timeit.default_timer() - start_time
        if t < args.warmup_steps:
            print("warmup")
        else:
            measurements.append(elapsed)



    print(f"mean: {statistics.mean(measurements) * 1000:.3f} ms")
    print(f"std:  {statistics.stdev(measurements) * 1000:.3f} ms")


if __name__ == "__main__":
    main()