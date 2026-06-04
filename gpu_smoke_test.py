import time

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment.")

    device = torch.device("cuda")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    n = 4096
    a = torch.randn((n, n), device=device)
    b = torch.randn((n, n), device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Matrix multiply: {tuple(c.shape)} in {elapsed:.3f}s")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
