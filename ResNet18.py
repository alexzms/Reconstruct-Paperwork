# ResNET 残差神经网络
# 主要的思想就是类似一个shortcut地方式将浅层神经网络的输出，加到深层神经网络的输出上
# 这样做的好处是防止梯度消失，因为在生成深度神经网络的时候由于链式法则，梯度会一路乘下去
# 而如果使用ResNet就使得grad_deep = grad_deep + grad_shallow，梯度不会消失
# 使得训练收敛更快，也能有效方式过拟合（深度网络的深层部分若冗余，则会自发构建identity map，能够fallback到浅层网络）

# 这里我先复现ResNet18，之后再研究一下论文中的bottleneck模型，复现一下ResNet50

import torch
import torch.nn as nn
