from __future__ import annotations

import argparse
import statistics

import torch

from mamba3.ops import mamba3_scan


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
    parser = argparse.ArgumentParser(description="Benchmark the chunked Mamba-3 SSD")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--state", type=int, default=128)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--training", action="store_true")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    dtype = getattr(torch, args.dtype)
    shape_qk = (args.batch, args.length, args.heads, args.rank, args.state)
    shape_value = (args.batch, args.length, args.heads, args.head_dim)
    q = torch.randn(shape_qk, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    value = torch.randn(shape_value, device="cuda", dtype=dtype)
    gate = torch.randn_like(value)
    dt = torch.rand(args.batch, args.length, args.heads, device="cuda") * 0.1
    adt = -torch.rand_like(dt) * dt
    trap = torch.randn_like(dt, dtype=dtype)
    D = torch.ones(args.heads, device="cuda", dtype=dtype)
    if args.rank > 1:
        projection_shape = (args.heads, args.rank, args.head_dim)
        mimo_x = torch.ones(projection_shape, device="cuda", dtype=dtype) / args.rank
        mimo_z = torch.ones_like(mimo_x)
        mimo_out = torch.ones_like(mimo_x) / args.rank
    else:
        mimo_x = mimo_z = mimo_out = None
    values = [q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out]
    if args.training:
        values = [item.requires_grad_(True) if item is not None else None for item in values]
    q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out = values

    def fused() -> None:
        output, _ = mamba3_scan(
            q,
            k,
            value,
            gate,
            adt,
            dt,
            trap,
            D,
            mimo_x=mimo_x,
            mimo_z=mimo_z,
            mimo_out=mimo_out,
            chunk_size=64 if args.rank == 1 else max(8, 64 // args.rank),
        )
        if args.training:
            output.float().square().mean().backward()
            for item in values:
                if item is not None:
                    item.grad = None

    torch.cuda.reset_peak_memory_stats()
    base_memory = torch.cuda.memory_allocated()
    latency = elapsed_ms(fused, warmup=10, iterations=args.iterations)
    tokens = args.batch * args.length
    peak_memory = torch.cuda.max_memory_allocated() - base_memory
    print(f"device: {torch.cuda.get_device_name()}")
    print(
        f"shape: B={args.batch}, L={args.length}, H={args.heads}, "
        f"P={args.head_dim}, N={args.state}, R={args.rank}"
    )
    print(f"mode/dtype: {'training' if args.training else 'inference'} / {args.dtype}")
    print(f"SSD: {latency:.3f} ms ({tokens / (latency / 1000):,.0f} tokens/s)")
    print(f"peak incremental memory: {peak_memory / (1024**2):.1f} MiB")


if __name__ == "__main__":
    main()
