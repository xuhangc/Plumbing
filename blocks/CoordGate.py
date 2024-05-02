import torch
import torch.nn as nn

"""CoordGate：高效计算卷积神经网络中的空间变化卷积
由于点扩散函数 （PSF），光学成像系统的分辨率固有限制，该函数对图像施加静态但空间变化的卷积。这种退化可以通过卷积神经网络（CNN）来解决，特别是通过去模糊技术。
然而，目前的解决方案在有效计算空间变化卷积方面面临一定的局限性。在本文中，我们提出了CoordGate，这是一种新颖的轻量级模块，它使用乘法门和坐标编码网络来高效计算CNN中的空间变化卷积。
CoordGate允许根据滤波器的空间位置对滤波器进行选择性放大或衰减，有效地充当本地连接的神经网络。


CoordGate模块是一个用于深度学习的PyTorch模块，设计用于在卷积神经网络中加入位置信息或映射信息，从而增强模型处理图像的能力。
这个模块允许通过不同的编码方式引入额外的空间信息，可以是位置编码、一个预定义的映射、或者通过双线性插值计算得到的映射。
这样的设计使得CoordGate模块非常适合处理需要考虑空间关系的任务，比如图像分类、分割或者其他视觉任务。

主要特点和应用：
灵活性：通过支持不同类型的编码，CoordGate模块可以适应各种任务和数据特性，提供了一种灵活的方式来融合空间信息。

增强的空间感知能力：将空间位置或自定义映射信息整合到卷积网络中，有助于提高模型对图像空间结构的理解，尤其是对于需要精确空间定位或分析的任务。

自适应特征调整：对于'map'和'bilinear'编码，CoordGate能够根据输入图像的特定特征动态调整权重，从而更好地处理图像中的变化和复杂性。
"""


class CoordGate(nn.Module):
    def __init__(self, enc_channels, out_channels, size: list = [256, 256], enctype='pos', **kwargs):
        super(CoordGate, self).__init__()
        '''
        type can be:'pos' - position encoding
                    'regularised' 
        '''

        self.enctype = enctype
        self.enc_channels = enc_channels

        if enctype == 'pos':

            encoding_layers = kwargs['encoding_layers']

            x_coord, y_coord = torch.linspace(-1, 1, int(size[0])), torch.linspace(-1, 1, int(size[1]))

            self.register_buffer('pos', torch.stack(torch.meshgrid((x_coord, y_coord), indexing='ij'), dim=-1).view(-1,
                                                                                                                    2))  # .to(device)

            self.encoder = nn.Sequential()
            for i in range(encoding_layers):
                if i == 0:
                    self.encoder.add_module('linear' + str(i), nn.Linear(2, enc_channels))
                else:
                    self.encoder.add_module('linear' + str(i), nn.Linear(enc_channels, enc_channels))

        elif (enctype == 'map') or (enctype == 'bilinear'):

            initialiser = kwargs['initialiser']

            if 'downsample' in kwargs.keys():
                self.sample = kwargs['downsample']
            else:
                self.sample = [1, 1]

            self.map = nn.Parameter(initialiser)

        self.conv = nn.Conv2d(enc_channels, out_channels, 1, padding='same')

        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        x is (bs,nc,nx,ny)
        '''
        if self.enctype == 'pos':

            gate = self.encoder(self.pos).view(1, x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
            gate = torch.nn.functional.relu(gate)  # ?
            x = self.conv(x * gate)
            return x


        elif self.enctype == 'map':

            map = self.relu(self.map).repeat_interleave(self.sample[0], dim=2).repeat_interleave(self.sample[1], dim=3)

            x = self.conv(x * map)
            return x

        elif self.enctype == 'bilinear':

            # if self.enc_channels == 9:
            map = create_bilinear_coeff_map_cart_3x3(self.map[:, 0:1], self.map[:, 1:2])
            # else:
            #     map = create_bilinear_coeff_map_cart_5x5(angles,distances)

            map = self.relu(map).repeat_interleave(self.sample[0], dim=2).repeat_interleave(self.sample[1], dim=3)

            x = self.conv(x * map)
            return x


def create_bilinear_coeff_map_cart_3x3(x_disp, y_disp):
    shape = x_disp.shape
    x_disp = x_disp.reshape(-1)
    y_disp = y_disp.reshape(-1)

    # Determine the quadrant based on the signs of the displacements
    primary_indices = torch.zeros_like(x_disp, dtype=torch.long)
    primary_indices[(x_disp >= 0) & (y_disp >= 0)] = 0  # Quadrant 1
    primary_indices[(x_disp < 0) & (y_disp >= 0)] = 2  # Quadrant 2
    primary_indices[(x_disp < 0) & (y_disp < 0)] = 4  # Quadrant 3
    primary_indices[(x_disp >= 0) & (y_disp < 0)] = 6  # Quadrant 4
    # Define the number of directions
    num_directions = 8

    # Compute the indices for the primary and secondary directions
    secondary_indices = ((primary_indices + 1) % num_directions).long()
    tertiary_indices = (primary_indices - 1).long()
    tertiary_indices[tertiary_indices < 0] = num_directions - 1

    x_disp = x_disp.abs()
    y_disp = y_disp.abs()

    coeffs = torch.zeros((x_disp.size(0), num_directions + 1), device=x_disp.device)
    batch_indices = torch.arange(x_disp.size(0), device=x_disp.device)

    coeffs[batch_indices, primary_indices] = (x_disp * y_disp)
    coeffs[batch_indices, secondary_indices] = x_disp * (1 - y_disp)
    coeffs[batch_indices, tertiary_indices] = (1 - x_disp) * y_disp
    coeffs[batch_indices, -1] = (1 - x_disp) * (1 - y_disp)

    swappers = (primary_indices == 0) | (primary_indices == 4)

    coeffs[batch_indices[swappers], secondary_indices[swappers]] = (1 - x_disp[swappers]) * y_disp[swappers]
    coeffs[batch_indices[swappers], tertiary_indices[swappers]] = x_disp[swappers] * (1 - y_disp[swappers])

    coeffs = coeffs.view(shape[0], shape[2], shape[3], num_directions + 1).permute(0, 3, 1, 2)
    reorderer = [0, 1, 2, 7, 8, 3, 6, 5, 4]

    return coeffs[:, reorderer, :, :]


if __name__ == '__main__':
    # 创建 CoordGate 模块的实例
    enc_channels = 32
    out_channels = 32
    size = [256, 256]
    enctype = 'pos'
    encoding_layers = 2
    initialiser = torch.rand((out_channels, 2))
    kwargs = {'encoding_layers': encoding_layers, 'initialiser': initialiser}
    block = CoordGate(enc_channels, out_channels, size, enctype, **kwargs)

    # 生成随机输入数据
    input_size = (1, enc_channels, size[0], size[1])
    input_data = torch.rand(input_size)

    # 对输入数据进行前向传播
    output = block(input_data)

    # 打印输入和输出数据的形状
    print("Input size:", input_data.size())
    print("Output size:", output.size())
