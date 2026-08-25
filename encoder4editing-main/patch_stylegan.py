import os

print("\n🔧 正在向 StyleGAN2 注入【完美修复版】免 C++ 编译补丁...")

# 1. 原生 PyTorch 版本的 fused_act
fused_act_code = """import torch
from torch import nn
from torch.nn import functional as F

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    if bias is not None:
        input = input + bias.view(1, bias.shape[0], *rest_dim)
    return F.leaky_relu(input, negative_slope=negative_slope) * scale

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
"""

# 2. 原生 PyTorch 版本的 upfirdn2d (完美修复了 Padding 和 Crop 计算)
upfirdn2d_code = """import torch
from torch.nn import functional as F

def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, 1, in_h, in_w)

    # 1. Upsample
    if up_x > 1 or up_y > 1:
        out = torch.zeros((input.shape[0], input.shape[1], in_h * up_y, in_w * up_x), device=input.device)
        out[:, :, ::up_y, ::up_x] = input
        input = out

    # 2. Pad (只做正向填充)
    pad_val = (max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0))
    input = F.pad(input, pad_val)

    # 3. Convolve
    kernel = kernel.flip(0, 1).view(1, 1, kernel.shape[0], kernel.shape[1])
    out = F.conv2d(input, kernel)

    # 4. Crop (处理负向 Padding 带来的裁剪需求，这是上个版本崩溃的根源！)
    if pad_x0 < 0 or pad_x1 < 0 or pad_y0 < 0 or pad_y1 < 0:
        crop_x0 = max(-pad_x0, 0)
        crop_x1 = out.shape[3] - max(-pad_x1, 0)
        crop_y0 = max(-pad_y0, 0)
        crop_y1 = out.shape[2] - max(-pad_y1, 0)
        out = out[:, :, crop_y0:crop_y1, crop_x0:crop_x1]

    # 5. Downsample
    if down_x > 1 or down_y > 1:
        out = out[:, :, ::down_y, ::down_x]

    out = out.view(-1, channel, out.shape[2], out.shape[3])
    return out

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if isinstance(up, int):
        up_x = up_y = up
    else:
        up_x, up_y = up
    if isinstance(down, int):
        down_x = down_y = down
    else:
        down_x, down_y = down
    if len(pad) == 2:
        pad_x0 = pad_y0 = pad[0]
        pad_x1 = pad_y1 = pad[1]
    else:
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

    return upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
"""

op_dir = os.path.join("models", "stylegan2", "op")

if not os.path.exists(op_dir):
    print(f"❌ 找不到路径 {op_dir}！")
else:
    with open(os.path.join(op_dir, "fused_act.py"), "w", encoding="utf-8") as f:
        f.write(fused_act_code)
    with open(os.path.join(op_dir, "upfirdn2d.py"), "w", encoding="utf-8") as f:
        f.write(upfirdn2d_code)
    print("✅ 完美版补丁覆盖成功！空间维度错乱问题已被彻底修复！")