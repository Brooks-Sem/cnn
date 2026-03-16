"""
CNN Inference Speed Benchmark
==============================
Measures inference throughput (images/sec) for classic CNN architectures
using torchvision pretrained models on CPU and (optionally) CUDA.

Usage:
    python benchmark.py                     # CPU only
    python benchmark.py --device cuda       # GPU
    python benchmark.py --batch 32 --runs 50
"""

import argparse
import time

import torch
import torchvision.models as models

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "AlexNet":       models.alexnet,
    "VGG-16":        models.vgg16,
    "GoogLeNet":     models.googlenet,
    "ResNet-50":     models.resnet50,
    "MobileNetV2":   models.mobilenet_v2,
    "EfficientNet-B0": models.efficientnet_b0,
}


def count_params(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def benchmark(
    model_name: str,
    batch_size: int = 16,
    input_size: tuple = (3, 224, 224),
    runs: int = 30,
    warmup: int = 5,
    device: str = "cpu",
) -> dict:
    """Run inference benchmark and return timing statistics."""
    factory = MODELS[model_name]
    model = factory(weights=None).to(device)
    model.eval()

    dummy = torch.randn(batch_size, *input_size, device=device)

    # Warm-up
    for _ in range(warmup):
        _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    avg_ms = (sum(latencies) / len(latencies)) * 1000
    throughput = batch_size / (avg_ms / 1000)
    params_m = count_params(model) / 1e6

    return {
        "model":       model_name,
        "params_M":    round(params_m, 2),
        "batch":       batch_size,
        "avg_ms":      round(avg_ms, 2),
        "throughput":  round(throughput, 1),
        "device":      device,
    }


def print_table(results: list[dict]) -> None:
    header = f"{'Model':<18} {'Params(M)':>10} {'Batch':>6} {'Avg(ms)':>9} {'Imgs/s':>9} {'Device':>6}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['model']:<18} {r['params_M']:>10.2f} {r['batch']:>6}"
            f" {r['avg_ms']:>9.2f} {r['throughput']:>9.1f} {r['device']:>6}"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN Inference Benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--batch",  type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--runs",   type=int, default=30,
                        help="Number of timed forward passes (default: 30)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup passes (default: 5)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Models to benchmark (default: all)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    print(f"\nCNN Inference Benchmark  |  device={args.device}  "
          f"batch={args.batch}  runs={args.runs}\n")

    results = []
    for name in args.models:
        print(f"  Benchmarking {name} ...", end=" ", flush=True)
        r = benchmark(
            name,
            batch_size=args.batch,
            runs=args.runs,
            warmup=args.warmup,
            device=args.device,
        )
        results.append(r)
        print(f"{r['avg_ms']:.1f} ms/batch  ({r['throughput']:.0f} imgs/s)")

    print()
    print_table(results)


if __name__ == "__main__":
    main()
