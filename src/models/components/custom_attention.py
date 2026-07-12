import torch
import fixed_offset_attention  # 你的 C++ 扩展

# import fixed_offset_similarity_cuda 

# ==========================================================================
# 1. Custom Op Wrapper
# ==========================================================================
class FixedOffsetSimilarityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, kernel_size, patch_size_h, patch_size_w):
        output = torch.empty(
            (query.shape[0], query.shape[1], query.shape[2], key.shape[1], kernel_size * kernel_size),
            device=query.device,
            dtype=query.dtype
        )
        fixed_offset_similarity_cuda.forward(
            query, key, output, kernel_size, patch_size_h, patch_size_w
        )
        ctx.save_for_backward(query, key)
        ctx.kernel_size = kernel_size
        ctx.patch_size_h = patch_size_h
        ctx.patch_size_w = patch_size_w
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key = ctx.saved_tensors
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        fixed_offset_similarity_cuda.backward(
            grad_output.contiguous(), query, key, grad_query, grad_key, 
            ctx.kernel_size, ctx.patch_size_h, ctx.patch_size_w
        )
        return grad_query, grad_key, None, None, None

def fixed_offset_similarity_cuda_op(query, key, kernel_size, patch_size):
    return FixedOffsetSimilarityFunction.apply(query, key, kernel_size, patch_size[0], patch_size[1])


class FixedOffsetAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, attn_weight, use_fp32_accum=None):
        # 1. 自动推断策略
        if use_fp32_accum is None:
            # 如果是半精度 (fp16/bf16)，默认开启 fp32 累加，保证数值稳定
            if value.dtype in [torch.float16, torch.bfloat16]:
                use_fp32_accum = True
            else:
                use_fp32_accum = False
        
        # 2. 确保连续内存 (非常重要！C++ 那边假设是连续的)
        value = value.contiguous()
        attn_weight = attn_weight.contiguous()

        ctx.save_for_backward(value, attn_weight)
        ctx.use_fp32_accum = use_fp32_accum
        
        output = fixed_offset_attention.forward(
            value, 
            attn_weight
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, attn_weight = ctx.saved_tensors
        
        # 确保梯度也是连续的
        grad_output = grad_output.contiguous()
        
        grad_value, grad_attn_weight = fixed_offset_attention.backward(
            grad_output, 
            value, 
            attn_weight, 
            ctx.use_fp32_accum  # 传入 forward 时决定的策略
        )
        
        # 返回参数必须和 forward 的参数数量一致
        # forward 有 3 个输入 (value, attn_weight, use_fp32_accum)
        # 所以 backward 返回 (grad_value, grad_attn, None)
        return grad_value, grad_attn_weight, None

# 最终对外的 API
def fixed_offset_attention_op(value, attn_weight, use_fp32_accum=None):
    # value: (B, L, C, H_feat, W_feat)
    # attn_weight: (B, H_attn, W_attn, L, num_offsets)
    return FixedOffsetAttentionFunction.apply(value, attn_weight, use_fp32_accum)
