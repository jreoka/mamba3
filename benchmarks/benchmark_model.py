from __future__ import annotations

import argparse
import statistics

import torch

import mamba3.ops as mamba3_ops
from mamba3 import Mamba3


def elapsed_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a complete Mamba-3 stack")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--mimo-rank", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()

    if args.cuda_graph:
        args.decode = True
    if args.decode and args.training:
        parser.error("--decode and --training are mutually exclusive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    dtype = getattr(torch, args.dtype)
    model = Mamba3(
        d_model=args.d_model,
        d_state=args.d_state,
        depth=args.depth,
        mimo_rank=args.mimo_rank,
    ).cuda()
    if not args.training:
        model = model.to(dtype=dtype)
    model.train(args.training)
    input_dtype = torch.float32 if args.training else dtype
    x = torch.randn(
        args.batch,
        args.length,
        args.d_model,
        device="cuda",
        dtype=input_dtype,
        requires_grad=args.training,
    )

    if args.decode:
        decoder = model.cuda_graph(args.batch) if args.cuda_graph else None

        @torch.inference_mode()
        def step() -> None:
            if decoder is None:
                cache = None
                for index in range(args.length):
                    _, cache = model.step(x[:, index : index + 1], cache)
            else:
                decoder.reset()
                decoder.validate()
                for index in range(args.length):
                    decoder(x[:, index : index + 1], validate=False)

    elif args.training:

        def step() -> None:
            with torch.autocast(
                "cuda", dtype=dtype, enabled=dtype != torch.float32
            ):
                loss = model(x).float().square().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            x.grad = None

    else:

        @torch.inference_mode()
        def step() -> None:
            model(x)

    torch.cuda.reset_peak_memory_stats()
    base_memory = torch.cuda.memory_allocated()
    latency = elapsed_ms(step, warmup=args.warmup, iterations=args.iterations)
    tokens = args.batch * args.length
    peak_memory = torch.cuda.max_memory_allocated() - base_memory

    print(f"device: {torch.cuda.get_device_name()}")
    print(
        f"shape: B={args.batch}, L={args.length}, D={args.d_model}, "
        f"N={args.d_state}, depth={args.depth}, rank={args.mimo_rank}"
    )
    if args.cuda_graph:
        backend = "cuda-graph"
    elif args.decode and mamba3_ops._TRITON_LAST_DISPATCH:
        backend = "triton"
    else:
        backend = "pytorch"
    print(
        f"mode/dtype/backend: "
        f"{'decode' if args.decode else 'training' if args.training else 'inference'} / "
        f"{args.dtype} / {backend}"
    )
    print(f"model: {latency:.3f} ms ({tokens / (latency / 1000):,.0f} tokens/s)")
    print(f"peak incremental memory: {peak_memory / (1024**2):.1f} MiB")


if __name__ == "__main__":
    main()
