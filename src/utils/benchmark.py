import time
from typing import Dict, Tuple

import torch


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
    }


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: torch.device,
    warmup_iters: int = 20,
    benchmark_iters: int = 100,
) -> Dict[str, float]:
    """Benchmark model inference latency and FPS using synthetic input.

    This utility isolates model speed from dataloader overhead. For CUDA devices,
    synchronization is used so the timing reflects actual GPU execution time.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    for _ in range(warmup_iters):
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    batch_size = input_shape[0]
    total_images = batch_size * benchmark_iters
    latency_ms_per_batch = (elapsed / benchmark_iters) * 1000.0
    latency_ms_per_image = (elapsed / total_images) * 1000.0
    fps = total_images / elapsed if elapsed > 0 else 0.0

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    return {
        "benchmark_batch_size": float(batch_size),
        "benchmark_iterations": float(benchmark_iters),
        "latency_ms_per_batch": float(latency_ms_per_batch),
        "latency_ms_per_image": float(latency_ms_per_image),
        "fps": float(fps),
        "peak_memory_mb": float(peak_memory_mb),
    }
