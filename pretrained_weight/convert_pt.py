import torch
from safetensors.torch import load_file

# 原文件路径
safetensors_path = "pretrained_weight/DA3-GIANT-1.1/model.safetensors"

# 保存路径
save_path = "pretrained_weight/DA3-GIANT-1.1/converted_model.pth"

# 加载 safetensors 为字典
state_dict = load_file(safetensors_path)

# for key in state_dict:
#     if "head.scratch.output_conv2_aux" in key:
#         print(key)
# import ipdb
# ipdb.set_trace()
# 只替换以 'model.' 开头的 key 前缀为 'net.'
new_state_dict = {
    key.replace("model.", "") if key.startswith("model.") else key: val
    for key, val in state_dict.items()
}

# 保存为 .pth 格式
torch.save(new_state_dict, save_path)

print(f"[✓] 安全替换完成，保存到：{save_path}")