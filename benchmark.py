import torch
import time

def benchmark_matmul(size=4096, iterations=100):
    """Benchmark AMD ROCm vs expected baseline"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iterations):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tflops = (2 * size**3 * iterations) / elapsed / 1e12
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Matrix: {size}x{size}, Iterations: {iterations}")
    print(f"Time: {elapsed:.2f}s, TFLOPS: {tflops:.1f}")
    return tflops

if __name__ == "__main__":
    benchmark_matmul()
