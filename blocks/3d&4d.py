import torch
from einops import rearrange
# to_3d
# 把4维的张量转换成3维的张量，输入形状(b,c,h,w), 输出形状(b,h*w,c)。
# (b,c,h,w)->(b,h*w,c)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


# to_4d
# 把3维的张量转换成4维的张量，输入形状(b,h*w,c), 输出形状(b,c,h,w)。
# (b,h*w,c)->(b,c,h,w)
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

#example
if __name__ == '__main__':
    # 定义一个4D张量
    tensor_4d = torch.randn(2, 3, 4, 5)  # 形状: (批大小=2, 通道数=3, 高度=4, 宽度=5)

    # 使用 to_3d 函数将4D张量转换为3D张量
    tensor_3d = to_3d(tensor_4d)  # 形状: (批大小=2, 高度*宽度=20, 通道数=3)

    # 使用 to_4d 函数将3D张量转换回4D张量
    # 注意: 需要提供高度和宽度信息
    height, width = 4, 5
    tensor_4d_restored = to_4d(tensor_3d, height, width)  # 形状: (批大小=2, 通道数=3, 高度=4, 宽度=5)

    # 验证形状
    print("原始4D张量形状:", tensor_4d.shape)
    print("重塑后的3D张量形状:", tensor_3d.shape)
    print("恢复后的4D张量形状:", tensor_4d_restored.shape)

    # 定义一个样本的3D张量
    tensor_3d = torch.randn(2, 4096, 3)  # 形状: (批大小=2, 高度*宽度=20, 通道数=3)

    # 推断高度和宽度
    length = tensor_3d.size(1)
    channels = tensor_3d.size(2)
    estimated_height = int(length ** 0.5)  # 假设高度和宽度相等
    estimated_width = int(length / estimated_height)

    # 使用 to_4d 函数将3D张量转换回4D张量
    tensor_4d_restored = to_4d(tensor_3d, estimated_height, estimated_width)  # 形状: (批大小=2, 通道数=3, 高度=估计高度, 宽度=估计宽度)

    # 验证形状
    print("原始3D张量形状:", tensor_3d.shape)
    print("恢复后的4D张量形状:", tensor_4d_restored.shape)
