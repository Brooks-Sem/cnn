"""
CNN Inference Speed Benchmark
Measures per-sample latency (ms) for classic CNN architectures using PyTorch.
Requires: torch torchvision
"""
import time
import torch
import torchvision.models as models

MODELS = {
    "AlexNet":     models.alexnet,
    "VGG-16":      models.vgg16,
    "ResNet-50":   models.resnet50,
    "MobileNetV2": models.mobilenet_v2,
    "EfficientB0": models.efficientnet_b0,
}

WARMUP = 10
RUNS = 50
BATCH = 1
INPUT_SIZE = (BATCH, 3, 224, 224)


def benchmark_model(name: str, model_fn) -> dict:
    model = model_fn(weights=None).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(INPUT_SIZE, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / RUNS * 1000  # ms per sample

    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {"name": name, "params_M": round(params, 1), "latency_ms": round(elapsed, 2)}


def main():
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}\n")
    print(f"{'Model':<14} {'Params (M)':>10} {'Latency (ms)':>14}")
    print("-" * 42)
    results = []
    for name, fn in MODELS.items():
        r = benchmark_model(name, fn)
        results.append(r)
        print(f"{r['name']:<14} {r['params_M']:>10.1f} {r['latency_ms']:>14.2f}")
    return results


if __name__ == "__main__":
    main()
