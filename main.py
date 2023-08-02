# GAN model experiment 2023-08-02
# GAN，一个天才的设计，对抗式生成网络
# 两个网络，一个生成器Generator，一个判别器Discriminator
# 两个网络相互对抗，最终达到一个平衡，生成器生成的图片越来越真实，判别器判别的越来越准确
# Genius

import torch
import torch.nn as nn
import torchvision

# 1 为channel，28为height，28为width
image_size = (1, 28, 28)
image_size_int = torch.prod(torch.tensor(image_size), dtype=torch.int32).item()
# 10个数字需要区分
mnist_discrimination_number = 10


class Generator(nn.Module):
    # TODO: in_dim的默认维度应该是什么呢?还得再好好想想
    def __init__(self, in_dim=mnist_discrimination_number, *args, **kwargs):
        # 这里的args是直接传的参数func(1)，kwargs是字典参数func(name="some guy")
        super(Generator, self).__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            # torch.prod计算image_size各个维度的乘积，.item()转换为int
            # 这一步其实就是把1024隐空间的向量转换为image，此为：Generator哒！
            nn.Linear(1024, image_size_int),
            # 将输出映射到-1:1中，图像灰度图
            nn.Tanh()
        )

    def forward(self, z):
        # z是一个[batch_size, torch.prod(image_size)]
        batch_size = z[0]
        output = self.model(z)
        assert (isinstance(output, torch.Tensor))
        # 这里的*image_size是解包，将image_size中的三个值传入reshape函数，把元组解包为三个参数，最终是(batch_size, 1, 28, 28)
        image_batches = output.reshape((batch_size, *image_size))

        return image_batches


class Discriminator(nn.Module):
    def __init__(self, in_dim=image_size_int, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # 二分类，输出0或1，也就是Discriminator在看到一张图片后，判断这张图片是真的还是假的
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        # image: [batch_size, *image_size]
        # 在PyTorch中，view() 和 reshape() 都用于改变张量的形状，但有一些细微的区别。
        #   内存连续性：reshape() 可能会返回一个与原始张量共享相同数据的不连续张量，而 view() 只在张量连续时才能使用。
        #   数据存储方式：当原始张量的数据存储是按行的时候，view() 可以保证返回的张量的每一行都是连续存储的。而 reshape() 返回的张量并不保证行的连续性。
        #   操作灵活性：reshape() 可以在不改变张量元素总数的情况下更灵活地改变张量的形状，而 view() 需要保持张量元素总数不变。
        flatten_image = image.view(image.shape[0], -1)  # 这里的-1就是把剩下的维度全部压进一个维度
        output_probability = self.model(flatten_image)

        # 这个probability就是一眼丁真中的有多“丁真”
        return output_probability


dataset = torchvision.datasets.MNIST(root="mnist_data", download=False, train=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize(image_size),
                                         torchvision.transforms.ToTensor()
                                     ]))

for i in range(10):
    img, label = dataset[i]
    print(img.shape, label)
