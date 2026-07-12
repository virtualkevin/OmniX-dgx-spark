import torch
from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch

# 配置随机输入形状参数
N, M, D = 1, 2, 4       # batch, num_heads, channels
Lq, L, P = 8, 2, 4      # num_query, num_levels, num_points
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])

torch.manual_seed(42)

def run_test(dtype, rtol=1e-3, atol=1e-4):
    print(f"\n==== Testing dtype: {dtype} ====")
    # 构造输入
    value = (torch.rand(N, S, M, D, dtype=dtype, device="cuda") * 0.01)
    sampling_locations = torch.rand(N, Lq, M, L, P, 2, dtype=dtype, device="cuda")
    attention_weights = torch.rand(N, Lq, M, L, P, dtype=dtype, device="cuda") + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 1

    # PyTorch参考输出
    out_pt = ms_deform_attn_core_pytorch(
        value.float(),     # PyTorch参考实现一般只支持 float
        shapes,
        sampling_locations.float(),
        attention_weights.float()
    ).detach().cpu()

    # CUDA扩展输出
    out_cuda = MSDeformAttnFunction.apply(
        value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    ).detach().cpu()

    # 比较误差
    if dtype != torch.float32:
        # bf16 跟 float32 baseline比较
        compare_to = out_pt.to(out_cuda.dtype)
    else:
        compare_to = out_pt

    fwd_ok = torch.allclose(out_cuda, compare_to, rtol=rtol, atol=atol)
    max_abs_err = (out_cuda - compare_to).abs().max().item()
    max_rel_err = ((out_cuda - compare_to).abs() / compare_to.abs().clamp_min(1e-9)).max().item()

    print(f"Forward match: {fwd_ok}")
    print(f"Max abs error: {max_abs_err:.3e}")
    print(f"Max rel error: {max_rel_err:.3e}")

if __name__ == "__main__":
    run_test(torch.float32, rtol=1e-3, atol=1e-4)
    run_test(torch.bfloat16, rtol=3e-2, atol=1e-2)