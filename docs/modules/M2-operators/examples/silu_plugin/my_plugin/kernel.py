"""Complete Chapter 24 standalone checks and three-baseline microbenchmark.

Requires an already compatible CUDA/PyTorch/Triton/vLLM environment. This file
has CPU-side syntax/contract checks in the repository; GPU JIT, engine dispatch,
numerical correctness and performance remain pending until run on that setup.
"""

import argparse
import json

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, d,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < d
    x1 = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    x2 = tl.load(x_ptr + row * x_row_stride + d + cols, mask=mask, other=0.0)
    x1_f32 = x1.to(tl.float32)
    silu = (x1_f32 * tl.sigmoid(x1_f32)).to(x1.dtype)
    tl.store(y_ptr + row * y_row_stride + cols, silu * x2, mask=mask)


def silu_and_mul_triton(x: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("last dimension must be even")
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("expected CUDA FP16/BF16 input")
    if not x.is_contiguous():
        raise ValueError("contiguous input is required; do not hide a copy in the timing")
    if BLOCK <= 0 or BLOCK & (BLOCK - 1) or BLOCK > 2**20:
        raise ValueError("BLOCK must be a supported power of two")
    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    if x.numel() == 0:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    x_2d = x.reshape(-1, x.shape[-1])
    m, n = x_2d.shape
    d = n // 2
    y = torch.empty((m, d), dtype=x.dtype, device=x.device)
    grid = (m, triton.cdiv(d, BLOCK))
    _silu_and_mul_kernel[grid](x_2d, y, x_2d.stride(0), y.stride(0), d, BLOCK=BLOCK)
    return y.reshape(output_shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=11008)
    parser.add_argument("--block", type=int, default=1024)
    args = parser.parse_args()
    if args.tokens <= 0 or args.hidden <= 0:
        parser.error("tokens and hidden must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable: GPU/engine checks remain pending")
    from vllm.model_executor.layers.activation import SiluAndMul
    import vllm
    import vllm._custom_ops  # noqa: F401  load the built-in CUDA op registrations

    print(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                      "vllm": vllm.__version__, "cuda_runtime": torch.version.cuda,
                      "device": torch.cuda.get_device_name(), "seed": 0, "BLOCK": args.block}))
    torch.manual_seed(0)
    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    else:
        print(json.dumps({"check": "bf16", "status": "pending", "reason": "device lacks BF16 support"}))
    for dtype in dtypes:
        for shape in ((1, 2), (3, 2000), (2, 3, 2046), (0, 64), (1, 0)):
            x = torch.randn(shape, dtype=dtype, device="cuda")
            if x.numel() >= 7:
                x.view(-1)[:7] = torch.tensor([0., -1., 1., -2., 2., -80., 80.],
                                             dtype=dtype, device="cuda")
            y = silu_and_mul_triton(x, BLOCK=args.block)
            ref = SiluAndMul.forward_native(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)
            error = (y.float() - ref.float()).abs().max().item() if y.numel() else 0.0
            print(json.dumps({"check": "standalone_vs_native", "status": "passed",
                              "shape": list(shape), "dtype": str(dtype), "max_abs": error}))
    if args.check_only:
        return

    def eager(t):
        return SiluAndMul.forward_native(t)

    def cpp_fused(t):
        out = torch.empty(t.shape[:-1] + (t.shape[-1] // 2,), dtype=t.dtype, device=t.device)
        torch.ops._C.silu_and_mul(out, t)
        return out

    x = torch.randn(args.tokens, 2 * args.hidden, dtype=torch.float16, device="cuda")
    for name, fn in (("eager", eager), ("vllm_cpp", cpp_fused),
                     ("triton", lambda t: silu_and_mul_triton(t, BLOCK=args.block))):
        actual, ref = fn(x), eager(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
        ms = triton.testing.do_bench(lambda: fn(x), warmup=25, rep=100, return_mode="median")
        print(json.dumps({"benchmark": name, "status": "measured", "ms": ms,
                          "shape": list(x.shape), "dtype": str(x.dtype), "statistic": "median",
                          "includes_output_allocation": True, "cache": "do_bench_cache_flush"}))


if __name__ == "__main__":
    main()
