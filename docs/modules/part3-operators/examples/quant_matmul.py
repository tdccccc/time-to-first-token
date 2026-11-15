"""Chapter 22: complete W8A8 and fused-dequant experiment entry.

Requires PyTorch, Triton and an NVIDIA CUDA device. Python syntax and the
FP32 quantization contract can be checked without those packages; GPU JIT,
correctness and performance are NOT claimed until this entry is run on a GPU.
Fixed tile configurations provide a baseline before adding autotune.
"""

import argparse
import json

import torch
import triton
import triton.language as tl


def quant_per_token(x):
    # Statistics, scales, division and rounding all stay in FP32.
    xf = x.float()
    amax = xf.abs().amax(dim=1)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(xf / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def quant_per_channel(w):
    wf = w.float()
    amax = wf.abs().amax(dim=0)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(wf / scale[None, :]).clamp(-127, 127).to(torch.int8)
    return q, scale


@triton.jit
def matmul_int8_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn,
                       stride_cm, stride_cn, scale_a_ptr, scale_b_ptr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    sa = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=0.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def matmul_fused_dequant_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                                stride_am, stride_ak, stride_bk, stride_bn,
                                stride_cm, stride_cn, scale_b_ptr,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        b_int8 = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0)
        b = (b_int8.to(tl.float32) * sb[None, :]).to(tl.float16)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_int8(a_q, b_q, scale_a, scale_b):
    if a_q.ndim != 2 or b_q.ndim != 2 or a_q.shape[1] != b_q.shape[0]:
        raise ValueError("expected A[M,K], B[K,N]")
    if a_q.dtype != torch.int8 or b_q.dtype != torch.int8:
        raise TypeError("W8A8 operands must be int8")
    M, K = a_q.shape
    N = b_q.shape[1]
    if K > (2**31 - 1) // (128 * 128):
        raise ValueError("K may overflow the int32 accumulator")
    if scale_a.shape != (M,) or scale_b.shape != (N,):
        raise ValueError("expected per-token scale_a[M], per-channel scale_b[N]")
    if not a_q.is_cuda or any(t.device != a_q.device for t in (b_q, scale_a, scale_b)):
        raise ValueError("all tensors must be on the same CUDA device")
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        raise TypeError("keep quantization scales in FP32")
    scale_a, scale_b = scale_a.contiguous(), scale_b.contiguous()
    c = torch.empty((M, N), dtype=torch.float16, device=a_q.device)
    if M == 0 or N == 0:
        return c
    if K == 0:
        return c.zero_()
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    matmul_int8_kernel[grid](
        a_q, b_q, c, M, N, K,
        a_q.stride(0), a_q.stride(1), b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1), scale_a, scale_b,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8,
        num_warps=4, num_stages=3,
    )
    return c


def matmul_fused_dequant(a, b_q, scale_b):
    if a.ndim != 2 or b_q.ndim != 2 or a.shape[1] != b_q.shape[0]:
        raise ValueError("expected A[M,K], B[K,N]")
    if a.dtype != torch.float16 or b_q.dtype != torch.int8 or scale_b.dtype != torch.float32:
        raise TypeError("expected FP16 activation, INT8 weight and FP32 scale")
    M, K = a.shape
    N = b_q.shape[1]
    if scale_b.shape != (N,):
        raise ValueError("expected per-channel scale_b[N]")
    if not a.is_cuda or any(t.device != a.device for t in (b_q, scale_b)):
        raise ValueError("all tensors must be on the same CUDA device")
    scale_b = scale_b.contiguous()
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    if M == 0 or N == 0:
        return c
    if K == 0:
        return c.zero_()
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    matmul_fused_dequant_kernel[grid](
        a, b_q, c, M, N, K,
        a.stride(0), a.stride(1), b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1), scale_b,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8,
        num_warps=4, num_stages=3,
    )
    return c


def check_scale_edges():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.tensor([[0., 0.], [1e-7, -1e-7], [1., -1.]], dtype=dtype)
        for quantize, values in ((quant_per_token, x), (quant_per_channel, x.T)):
            q, scale = quantize(values)
            assert scale.dtype == torch.float32
            assert bool(torch.isfinite(scale).all() and (scale > 0).all())
            assert bool(torch.isfinite(q.float()).all())
            zero_slice = q[0] if quantize is quant_per_token else q[:, 0]
            assert bool((zero_slice == 0).all())
    print(json.dumps({"check": "scale_edges_cpu", "status": "passed"}))


def validate_case(a, b):
    aq, sa = quant_per_token(a)
    bq, sb = quant_per_channel(b)
    int8_out = matmul_int8(aq, bq, sa, sb)
    fused_out = matmul_fused_dequant(a, bq, sb)
    torch.cuda.synchronize()
    ac, bc, aqc, bqc = (t.cpu() for t in (a, b, aq, bq))
    sac, sbc = sa.cpu(), sb.cpu()
    b_dequant = (bqc.float() * sbc[None, :]).half()
    M, K = a.shape
    N = b.shape[1]
    full = M * N * K <= 64_000_000
    if full:
        int_sum = aqc.to(torch.int64) @ bqc.to(torch.int64)
        int_ref = (int_sum.float() * sac[:, None] * sbc[None, :]).half()
        fused_ref = (ac.double() @ b_dequant.double()).half()
        original_ref = (ac.double() @ bc.double()).half()
        int_actual, fused_actual = int8_out.cpu(), fused_out.cpu()
    else:
        generator = torch.Generator().manual_seed(7)
        rows = torch.cat((torch.tensor([0, M - 1]), torch.randint(M, (64,), generator=generator)))
        cols = torch.cat((torch.tensor([0, N - 1]), torch.randint(N, (64,), generator=generator)))
        int_sum = (aqc[rows].to(torch.int64) * bqc[:, cols].T.to(torch.int64)).sum(dim=1)
        int_ref = (int_sum.float() * sac[rows] * sbc[cols]).half()
        fused_ref = (ac[rows].double() * b_dequant[:, cols].T.double()).sum(dim=1).half()
        original_ref = (ac[rows].double() * bc[:, cols].T.double()).sum(dim=1).half()
        int_actual, fused_actual = int8_out.cpu()[rows, cols], fused_out.cpu()[rows, cols]
    assert bool(torch.isfinite(int8_out).all() and torch.isfinite(fused_out).all())
    # Kernel errors are checked against the SAME quantized operands, not the
    # original FP16 model. Algorithmic quantization error is reported separately.
    torch.testing.assert_close(int_actual, int_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(fused_actual, fused_ref, atol=1e-2, rtol=1e-3)
    quant_error = (int_actual.float() - original_ref.float()).abs()
    print(json.dumps({"check": "gpu_kernel_reference", "shape": [M, N, K],
                      "status": "passed", "coverage": "full" if full else "sampled_66",
                      "int8_max_abs": (int_actual.float() - int_ref.float()).abs().max().item(),
                      "fused_max_abs": (fused_actual.float() - fused_ref.float()).abs().max().item(),
                      "quantization_max_abs": quant_error.max().item(),
                      "quantization_mean_scaled_error": (quant_error / original_ref.float().abs().clamp_min(1)).mean().item()}))
    return aq, bq, sa, sb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("dimensions must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable: GPU checks/performance remain pending")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                      "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(),
                      "dtype": "fp16", "seed": 0, "configuration": [64, 64, 32, 4, 3]}))
    check_scale_edges()
    # Every dimension has a non-multiple case; a zero row/column is also tested.
    for M, N, K in ((1, 17, 35), (32, 64, 64), (65, 63, 33)):
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        a[0] = 0
        b[:, 0] = 0
        validate_case(a, b)
    a = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16)
    b = torch.randn(args.k, args.n, device="cuda", dtype=torch.float16)
    aq, bq, sa, sb = validate_case(a, b)
    if args.check_only:
        return

    def dynamic_int8():
        activation, scale = quant_per_token(a)
        return matmul_int8(activation, bq, scale, sb)

    def two_step():
        dequant_weight = (bq.float() * sb[None, :]).half()
        return a @ dequant_weight

    for name, fn in (
        ("fp16_matmul", lambda: a @ b),
        ("int8_wrapper_only", lambda: matmul_int8(aq, bq, sa, sb)),
        ("dynamic_quant_plus_int8", dynamic_int8),
        ("fused_dequant", lambda: matmul_fused_dequant(a, bq, sb)),
        ("two_step_dequant_plus_matmul", two_step),
    ):
        fn()  # Finish JIT before the measured repetitions.
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
        print(json.dumps({"benchmark": name, "status": "measured", "ms": ms,
                          "shape": [args.m, args.n, args.k], "statistic": "median",
                          "cache": "do_bench_cache_flush", "includes_output_allocation": True}))


if __name__ == "__main__":
    main()
