# GAN model experiment

import torch
import torch.nn as nn
import torchvision


class Generator(nn.Module):
    pass


class Discriminator(nn.Module):
    pass


dataset = torchvision.datasets.MNIST(root="mnist_data", download=True, train=True)
print(len(dataset))