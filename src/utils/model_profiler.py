import time
import torch


def profile_model(model, input_size=(1,3,360,480), device='cpu', warmup=10, runs=50):
    model=model.to(device).eval()
    x=torch.randn(*input_size, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            _=model(x)

    if device=='cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start=time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _=model(x)

    if device=='cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem=torch.cuda.max_memory_allocated()/(1024**2)
    else:
        peak_mem=0.0

    elapsed=time.perf_counter()-start
    latency=(elapsed/runs)*1000
    fps=(runs*input_size[0])/elapsed

    params=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'parameters':params,
        'trainable_parameters':trainable,
        'latency_ms':latency,
        'fps':fps,
        'peak_memory_mb':peak_mem,
    }
