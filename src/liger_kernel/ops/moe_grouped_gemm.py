import torch

import triton
import triton.language as tl


@triton.jit
def _moe_grouped_gemm_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_size: tl.constexpr,
):
    pid = tl.program_id(0)
    # TODO: persistent loop over tiles, grouped GEMM per expert


def moe_grouped_gemm_forward(*args, **kwargs):
    """Grouped GEMM over experts (Triton)."""
    raise NotImplementedError


def moe_grouped_gemm_backward(*args, **kwargs):
    """backward for MoE grouped GEMM."""
    raise NotImplementedError
