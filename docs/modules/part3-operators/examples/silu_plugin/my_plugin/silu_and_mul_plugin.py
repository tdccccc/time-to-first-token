"""NVIDIA CustomOp example: registration is not enablement or runtime proof."""

import torch
from vllm.model_executor.layers.activation import SiluAndMul

from .kernel import silu_and_mul_triton


@SiluAndMul.register_oot
class TritonSiluAndMul(SiluAndMul):
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul_triton(x)

    # Keep the parent's forward_native and constructor behavior. On an OOT
    # hardware platform this CUDA method is not a substitute for forward_oot.
