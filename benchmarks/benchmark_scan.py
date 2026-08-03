from __future__ import annotations

import argparse
import statistics

import torch

from mamba3.ops import load_cuda_extension, selective_scan


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
    parser = argparse.ArgumentParser(description="Benchmark the fused selective scan")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--state", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if load_cuda_extension(verbose=True) is None:
        raise SystemExit("CUDA extension could not be built")

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.batch, args.length, args.channels, device=device, dtype=dtype)
    dt = torch.rand_like(x) * 0.05
    A = -torch.rand(args.channels, args.state, device=device)
    B = torch.randn(args.batch, args.length, args.state, device=device, dtype=dtype)
    C = torch.randn_like(B)
    D = torch.ones(args.channels, device=device)
    z = torch.randn_like(x)
    values = (x, dt, A, B, C, D, z)
    if args.training:
        values = tuple(value.detach().requires_grad_(True) for value in values)

    def fused():
        y = selective_scan(*values, use_cuda_kernel=True)
        if args.training:
            y.float().square().mean().backward()
            for value in values:
                value.grad = None

    torch.cuda.reset_peak_memory_stats()
    base_memory = torch.cuda.memory_allocated()
    latency = elapsed_ms(fused, warmup=10, iterations=args.iterations)
    tokens = args.batch * args.length
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"shape: B={args.batch}, L={args.length}, H={args.channels}, N={args.state}")
    print(f"mode/dtype: {'training' if args.training else 'inference'} / {args.dtype}")
    print(f"fused scan: {latency:.3f} ms ({tokens / (latency / 1000):,.0f} tokens/s)")
    peak_memory = torch.cuda.max_memory_allocated() - base_memory
    print(f"peak incremental memory: {peak_memory / (1024 ** 2):.1f} MiB")


if __name__ == "__main__":
    main()
