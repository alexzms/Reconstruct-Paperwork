# ResNET 残差神经网络 model experiment 2023-08-05
# 主要的思想就是类似用shortcut连接的方式将浅层神经网络的输出，加到深层神经网络的输出上
# 这样做的好处是防止梯度消失，因为在生成深度神经网络的时候由于链式法则，梯度会一路乘下去 grad_deep = grad_add * grad_shallow
# 而如果使用ResNet就使得grad_deep = grad_deep(=grad_add*grad_shallow) + grad_shallow，梯度不会消失(grad_shallow通常较大)
# 使得训练收敛更快，也能有效方式过拟合（深度网络的深层部分若冗余，则会自发构建identity map，能够fallback到浅层网络）


import torch
import torch.nn as nn
from torchsummary import summary  # summary(model, input_shape)

default_expansion = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ResNet50的ResBlock其实非常简单
# input -> 1x1 conv sampler -> 3x3 conv -> 1x1 expansion conv -> output
class ResBlock(nn.Module):
    """
    out_channel: 实际上真实的out_channel是out_channel *4
    """

    def __init__(self, in_channels: int, out_channels: int, expansion=default_expansion, nn_identity_downsample=None,
                 stride=1, *args, **kwargs):
        super(ResBlock, self).__init__(*args, **kwargs)
        assert isinstance(expansion, int)
        if nn_identity_downsample is not None:
            assert isinstance(nn_identity_downsample, nn.Module)
        if out_channels % expansion != 0:
            print("Resblock init failed: out_channel % expansion != 0")
        middle_out_channels = out_channels // expansion
        # self.expansion：这是一个扩展系数，目的是在最后一个1x1卷积层中扩增特征图的深度。
        # 在ResNet50以上的模型中，self.expansion通常设置为4。
        self.expansion = expansion
        # self.conv1、self.conv2、self.conv3：这三个是卷积层
        # 在ResNet的残差块中分别用于降维、提取特征、恢复维度。
        # 它们的核大小分别为1x1、3x3、1x1，是ResNet的瓶颈结构设计。
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=middle_out_channels, kernel_size=1, stride=1,
                               padding=0)
        # self.bn1、self.bn2、self.bn3：这三个是批归一化（Batch Normalization）层
        # 作用是提高训练速度，抑制过拟合，通常跟在卷积层后面。
        self.bn1 = nn.BatchNorm2d(middle_out_channels)
        # 这个是3x3的conv网络
        # stride：卷积核移动的步长，例如1、2等。
        # 这是可以调整的参数，一般而言，步长为1意味着卷积核每次移动一个像素距离，而步长为2则每次移动两个像素距离
        # 如果说这里stride=2，那么就需要identity_downsample起作用了（此后，图像大小都会减半）
        self.conv2 = nn.Conv2d(in_channels=middle_out_channels, out_channels=middle_out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_out_channels)
        # expansion 1X1，扩展特征图的深度
        # padding：卷积操作中的填充数，是在输入特征图周围填充0的层数，主要用于控制输出特征图的空间尺寸。
        # 例如，在使用kernel_size=3的卷积核时，若希望输出的特征图尺寸不变，则需要设置padding=1。
        self.conv3 = nn.Conv2d(in_channels=middle_out_channels, out_channels=out_channels, kernel_size=1, stride=1,
                               padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # identity downsample method（如果conv3输出和一开始的identity维度数不一样，就downsample）
        # 一个可能的写法：
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride),
        #     nn.BatchNorm2d(out_channel * self.expansion)
        # )
        self.nn_identity_downsample = nn_identity_downsample

    def forward(self, x):
        identity = x
        # 1x1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # 最后这一层先不调用激活函数，因为我们要插入residual，也就是加入identity这个shortcut
        x = self.bn3(self.conv3(x))

        # 如果需要downsample，就操作一下。一般来说会是用conv2d扩展self.expansion深度
        if self.nn_identity_downsample is not None:
            identity = self.nn_identity_downsample(identity)

        # residual
        x += identity

        return x


# 因为Res50, 101, 152在ResLayer的构件上是同构的，所以我们只需要让ResNet有layers=[..]这个可变参数就可以生成Res50,101,152
class ResNet(nn.Module):
    """
    resblock: 就是ResBlock的实例
    layers: 对于ResNet50，layers是[3,4,6,3]，详见ResNet-Paper(Page 5)
    image_channels: 对于RGB图像，就是3；对于灰阶图像就是1
    num_classes: 就是最后要分几个类别
    """

    def __init__(self, resblock, layers, image_channels, num_classes, expansion=default_expansion, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self.expansion = expansion
        # 不论是哪一种ResNet，一开始进入ResBlock之前的in_channel都是7x7 conv 64 stride=2，详见ResNet-Paper(Page 5)
        self.in_channels = 64
        # 之所以padding是3，因为当stride=1的时候padding=(k-1)/2可以保持图像大小不变，而当stride=2的时候相当于原图刚好缩小一半
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=self.in_channels, kernel_size=7, stride=2,
                               padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        # 3x3 maxpool stride=2 详见ResNet-Paper(Page 5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResLayers
        self.reslayer1 = self._make_layer(resblock=resblock, num_resblock=layers[0],
                                          out_channels=64 * self.expansion, stride=1)
        self.reslayer2 = self._make_layer(resblock=resblock, num_resblock=layers[1],
                                          out_channels=128 * self.expansion, stride=2)
        self.reslayer3 = self._make_layer(resblock=resblock, num_resblock=layers[2],
                                          out_channels=256 * self.expansion, stride=2)
        self.reslayer4 = self._make_layer(resblock=resblock, num_resblock=layers[3],
                                          out_channels=512 * self.expansion, stride=2)  # -> 512*expansion个图
        # nn.AdaptiveAvgPool2d((1, 1)) 是一个自适应的平均池化层。它会把每个通道的输入特征图调整为（1,1）大小
        # 实际上就是计算每个通道上特征图的平均值。这使得无论输入特征图的尺寸大小如何，都会输出固定尺寸的输出，便于最后的分类和回归任务。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # -> 512*expansion个1x1的图
        self.fc = nn.Linear(in_features=512 * self.expansion, out_features=num_classes)

    def forward(self, x):
        # conv1: 7x7 conv 64 stride 2
        x = self.relu(self.bn1(self.conv1(x)))  # -> 112 x 112 (64)
        x = self.maxpool(x)
        # conv2_x: ResBlock 1, (64->256)x3
        x = self.reslayer1(x)  # -> 56 x 56 (256)
        # conv3_x: ResBlock 2, (128->512)x4
        x = self.reslayer2(x)  # -> 28 x 28 (512)
        # conv4_x: ResBlock 3, (256->1024)x6
        x = self.reslayer3(x)  # -> 14 x 14 (1024)
        # conv5_x: ResBlock 4, (512->2048)x3
        x = self.reslayer4(x)  # -> 7 x 7 (2048)
        # Average Pooling & Fully Connected Layer
        x = self.avgpool(x)
        # 把每个1x1的值提取出来，flatten成一个列表
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, resblock, num_resblock, out_channels, stride):
        identity_downsample = None
        layers = []

        # or之前：输入图像层的大小不一致。和ResBlock里一样，如果stride不是1，就需要identity_downsample
        # or之后：输入图像层的数量不一致。ResNet的layer假设是第一种([3,4,5,6])，ResBlock要自己连自己连成3层:Res1->Res2->Res3
        #   那么在第一次Res1的时候，64层的input要加到256的output上面，这个时候就需要identity_downsample
        #   而在之后Res2，Res3，input和output都是64*4=256，所以不需要identity_downsample
        #   好聪明的设计！
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1,
                          stride=stride, padding=0),
                nn.BatchNorm2d(num_features=out_channels)
            )
        # Res1: in->out (out通常会是in*4), 所以此时in!=out, 需要identity_downsample
        layers.append(resblock(in_channels=self.in_channels, out_channels=out_channels,
                               nn_identity_downsample=identity_downsample, stride=stride))

        self.in_channels = out_channels
        for i in range(num_resblock - 1):
            # Res2,3... in->out (in=out)
            layers.append(resblock(in_channels=self.in_channels, out_channels=out_channels))

        return nn.Sequential(*layers)


def ResNet50(image_channels=3, num_classes=1000):
    return ResNet(resblock=ResBlock, layers=[3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes)


def ResNet101(image_channels=3, num_classes=1000):
    return ResNet(resblock=ResBlock, layers=[3, 4, 23, 3], image_channels=image_channels, num_classes=num_classes)


def ResNet152(image_channels=3, num_classes=1000):
    return ResNet(resblock=ResBlock, layers=[3, 8, 36, 3], image_channels=image_channels, num_classes=num_classes)


def test_network():
    net = ResNet152().to(device)
    # (number_images, image_channels, height, width)
    x = torch.randn((10, 3, 300, 224)).to(device)
    y = net(x)
    print(y.shape)


test_network()

# train
