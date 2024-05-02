import torch
import torch.nn as nn
import torch.nn.functional as F
"""
初始化 (__init__):
prompt_dim, prompt_len, prompt_size 和 lin_dim 是用于定义提示和线性层维度的参数。
prompt_param: 这是一个表示提示的参数张量，初始化为随机值。其形状为 (1, prompt_len, prompt_dim, prompt_size, prompt_size)。
linear_layer: 一个从 lin_dim 到 prompt_len 的线性层。
conv3x3: 一个2D卷积层，使用3x3的核，prompt_dim个输入通道和prompt_dim个输出通道。此层没有偏置项 (bias=False)。

前向传播 (forward):
x: 输入张量，形状为 (B, C, H, W)，其中 B 是批量大小，C 是通道数，H 和 W 是输入特征图的高度和宽度。
emb: 计算输入特征图 x 沿空间维度（高度和宽度）的均值，得到形状为 (B, C) 的张量。
prompt_weights: 对 emb 进行线性变换，然后沿第1维应用 softmax 激活，得到形状为 (B, prompt_len) 的张量。这表示分配给每个提示的权重。
prompt: 根据 prompt_weights 计算提示的加权和。根据输入特征图插值得到提示。在求和和插值后，此张量的形状为 (B, prompt_dim, H, W)。
conv3x3(prompt): 对插值后的提示张量应用3x3卷积，得到输出提示张量。
"""
#Prompt Generation Module
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt
if __name__ == '__main__':
    # 创建一个模块实例
    prompt_gen_block = PromptGenBlock(prompt_dim=3, prompt_len=4, prompt_size=96, lin_dim=3)#修改这里来对齐

    # 创建一个随机输入张量
    input_tensor = torch.randn(4, 3, 64, 64)  # 示例输入形状为 (B, C, H, W)，这里使用了 (4, 3, 64, 64)

    # 前向传播
    output_tensor = prompt_gen_block(input_tensor)

    # 打印输入和输出形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    """
    在代码 prompt_gen_block = PromptGenBlock(prompt_dim=3, prompt_len=4, prompt_size=96, lin_dim=3) 中，指定了 PromptGenBlock 类的初始化参数。这些参数代表了模型的结构和特性，具体含义如下：

    prompt_dim: 提示的特征维度。这个参数决定了提示张量中每个位置的特征向量的维度。在初始化提示参数时，每个位置的特征向量的维度将是 prompt_dim。
    prompt_len: 提示的长度或数量。这个参数决定了提示张量中包含的提示的个数。在初始化提示参数时，将会生成 prompt_len 个不同的提示。
    prompt_size: 提示张量的空间尺寸。这个参数决定了提示张量的高度和宽度。在初始化提示参数时，提示张量的形状将是 (prompt_len, prompt_dim, prompt_size, prompt_size)。
    lin_dim: 线性层的输出维度。这个参数决定了输入嵌入向量经过线性层后的输出维度。
    因此，通过指定这些参数，你可以控制提示生成模块的提示特征的维度、数量、空间尺寸以及线性层的输出维度，从而根据具体任务的要求来调整模型的结构。
    """