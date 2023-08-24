# ResNET 残差神经网络 model experiment 2023-08-05
# 主要的思想就是类似用shortcut连接的方式将浅层神经网络的输出，加到深层神经网络的输出上
# 这样做的好处是防止梯度消失，因为在生成深度神经网络的时候由于链式法则，梯度会一路乘下去 grad_deep = grad_add * grad_shallow
# 而如果使用ResNet就使得grad_deep = grad_deep(=grad_add*grad_shallow) + grad_shallow，梯度不会消失(grad_shallow通常较大)
# 使得训练收敛更快，也能有效方式过拟合（深度网络的深层部分若冗余，则会自发构建identity map，能够fallback到浅层网络）
import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torch.cuda.amp
import torchvision.transforms
import timm.utils
from torchsummary import summary  # summary(model, input_shape)

default_expansion = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_size = 224
img_channels = 3
resnet_ver = 50
dataset_path = './ResNet_data/fruit_vegetable_data_224'
log_path = './ResNet_data/logs'
log_step = 5
batch_size = 100
# 会被构建dataset的时候计算出来
num_categories = None
# num_workers参数决定了有多少个子进程被用来加载数据。
# 如果num_workers=0，则数据将在主进程中加载，这可能会阻塞主进程的其他操作。
# 如果num_workers大于0，数据加载就会由这些子进程完成，可以在后台并行进行，大大提高了数据加载的速度。
num_workers = 10
# pin_memory是一个布尔类型的参数，如果设置为True，DataLoader会在返回之前，将张量复制到CUDA固定内存(也称作pinned内存)中。
# 默认情况下，当你把一个在CPU上的Tensor送入GPU的时候，PyTorch总是会首先把它复制到一个pinned内存的buffer
# 然后CUDA才从那里异步的把它发送到GPU。如果你的数据在一开始就已经保存在pinned内存中了，那么这个额外的复制就可以被避免
# 从而可能会加速模型训练。
pin_memory = True
learn_rate = 0.01
# 就是IML课里学的例如Lasso或者Ridge回归的L2正则项的lambda系数
weight_decay = 0.001
num_epoch = 300


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


def ResNet50(image_channels=3, num_classes=1000) -> ResNet:
    return ResNet(resblock=ResBlock, layers=[3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes)


def ResNet101(image_channels=3, num_classes=1000) -> ResNet:
    return ResNet(resblock=ResBlock, layers=[3, 4, 23, 3], image_channels=image_channels, num_classes=num_classes)


def ResNet152(image_channels=3, num_classes=1000) -> ResNet:
    return ResNet(resblock=ResBlock, layers=[3, 8, 36, 3], image_channels=image_channels, num_classes=num_classes)


def ResNetCustomize(image_channels=3, num_classes=1000, layers=None) -> ResNet:
    if layers is None:
        layers = [3, 8, 36, 3]
    return ResNet(resblock=ResBlock, layers=layers, image_channels=image_channels, num_classes=num_classes)


def test_network():
    net = ResNet152().to(device)
    # (number_images, image_channels, height, width)
    x = torch.randn((10, 3, 300, 224)).to(device)
    y = net(x)
    print(y.shape)


# 训练部分
def build_transform(is_train) -> torchvision.transforms:
    """
    针对不同的运算模式，构建不同的图像预处理transform
    """
    transform = None
    if is_train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize_size),
            # 训练的时候，图像数据增强
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            # 这个函数将会生成一个四点透视变换，并用随机生成的值调整图像的四个角。
            # 四点透视变换是将图像的四个角移动到新的位置以创建一个新的视角。
            # 这种方法主要用于图像增强和数据增广，通过提供多样化的視觉效果来增强模型的泛化能力。
            torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1),
            # kernel_size 参数定义了高斯卷积核的大小，其指定了运算窗口的大小。
            # 它可能是一个整数或一个二元素元组。如果它是一个整数，那么水平和垂直方向上会有着同样的核大小。
            # 如果它是一个元组，那么第一个元素是水平方向上的核大小，第二个元素是垂直方向上的核大小。
            # 此处提供的是 kernel_size=(5, 9)，意味着高斯核的大小在水平方向是5，在垂直方向是9。
            # sigma 参数定义了高斯滤波器中的标准差值，决定了滤波器或模糊的强度，可以是一个单一的浮点数或一个二元素元组
            # 分别代表最小和最大可以使用的标准偏差。对于给定的图像，标准偏差会在配置的范围内随机选取。
            # 在此处，它为 (0.1, 5)，意味着标准偏差的值会在0.1和5之间随机选取。
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            torchvision.transforms.ToTensor()
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize_size),
            torchvision.transforms.ToTensor()
        ])
    return transform


def build_dataset(is_train, dataset_folder) -> torchvision.datasets.ImageFolder:
    """
    is_train: 是否是训练集, 用来设置Transform是否需要数据增强
    dataset_folder: 'train' 或 'test' 或 'validation'
    """
    global num_categories
    transform = build_transform(is_train=is_train)
    path = os.path.join(dataset_path, dataset_folder)
    # 由于文件夹层级结构非常标准./category/n.png，直接用ImageFolder库
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    info_dataset = dataset.find_classes(directory=path)
    num_categories = len(info_dataset[0])
    print(f"build_dataset({dataset_folder}): finding {len(info_dataset[0])} categories in {path}")
    print(f"build_dataset({dataset_folder}): {info_dataset[1]}")
    return dataset


def build_dataloader(is_train, dataset) -> torch.utils.data.DataLoader:
    """
    is_train: 设定是随机取样还是逐个取样
    """
    dataloader = None
    if is_train:
        sampler = torch.utils.data.RandomSampler(data_source=dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, sampler=sampler, batch_size=batch_size,
            # num_workers参数决定了有多少个子进程被用来加载数据。
            # 如果num_workers=0，则数据将在主进程中加载，这可能会阻塞主进程的其他操作。
            # 如果num_workers大于0，数据加载就会由这些子进程完成，可以在后台并行进行，大大提高了数据加载的速度。
            num_workers=num_workers,
            # pin_memory是一个布尔类型的参数，如果设置为True，DataLoader会在返回之前，将张量复制到CUDA固定内存(也称作pinned内存)中。
            # 默认情况下，当你把一个在CPU上的Tensor送入GPU的时候，PyTorch总是会首先把它复制到一个pinned内存的buffer
            # 然后CUDA才从那里异步的把它发送到GPU。如果你的数据在一开始就已经保存在pinned内存中了，那么这个额外的复制就可以被避免
            # 从而可能会加速模型训练。
            pin_memory=pin_memory,
            drop_last=True
        )
    else:
        sampler = torch.utils.data.SequentialSampler(data_source=dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True
        )
    return dataloader


def create_model(resnet=resnet_ver, layers=None):
    model = None
    if num_categories is None:
        print("fatal error: calling create_model while dataset is not built")
        exit(1)
    assert isinstance(num_categories, int)
    if layers is not None:
        model = ResNetCustomize(image_channels=img_channels, num_classes=num_categories)
    if resnet == 50:
        model = ResNet50(image_channels=img_channels, num_classes=num_categories)
    elif resnet == 101:
        model = ResNet101(image_channels=img_channels, num_classes=num_categories)
    elif resnet == 152:
        model = ResNet152(image_channels=img_channels, num_classes=num_categories)
    return model


# 不需要计算梯度
@torch.no_grad()
def validation(model, dataloader_validation, loss_func):
    """
    topk: 对于每个样本，从模型的输出中选择概率最高的前 k 个类别。
    检查真实标签是否在这 k 个类别中。如果是，则该样本被视为正确分类；否则，视为错误分类。
    这个函数会计算top1和top5正确率
    """
    accumulate_acc1 = 0
    accumulate_acc5 = 0
    accumulate_loss = 0
    accumulate_count = 0
    for validation_index, validation_sample in enumerate(dataloader_validation):
        accumulate_count += 1
        validation_image, validation_gt_label = validation_sample
        print(validation_gt_label)
        torchvision.utils.save_image(tensor=validation_image, fp='./test.png')
        validation_image = validation_image.to(device, non_blocking=True)
        validation_gt_label = validation_gt_label.to(device, non_blocking=True)
        # dim=-1代表在最后一个维度上进行计算，在这个计算中，因为validation_resnet_label.shape=(batch_size, categories_num)
        # 所以dim=-1和dim=1是一样的，并且acc1和acc5都是精确度百分比
        validation_resnet_label = nn.functional.softmax(model(x=validation_image), dim=-1)
        loss = loss_func(validation_resnet_label, validation_gt_label)
        # output: 模型的预测输出。通常是一个张量，表示模型的原始输出。该张量的形状应为 (batch_size, num_classes)
        #   其中 batch_size 是批处理大小，num_classes 是分类的类别数。
        # target: 真实的标签。通常是一个张量，表示每个样本的真实标签。该张量的形状应与 output 的形状相同。
        # topk: 一个元组，用于指定计算前几个最高概率的类别。例如，topk=(1,) 表示计算最高概率的类别，topk=(1, 5)
        #   表示同时计算最高概率和前五个最高概率的类别。
        # 例如说batch_size=3，并且我们选取top1，对于第一个样例输入分类对了，第二个分类错了，第三个分类对了，
        #   timm.utils.accuracy会输出0.667
        # 代码例子：tensor1 = torch.Tensor([[1, 2, 3], [2,1,3]]); tensor2 = torch.Tensor([2, 1]); 第一个分类对了(2)，第二个错
        # acc1, = timm.utils.accuracy(output=tensor1, target=tensor2, topk=(1, )) --> acc1.item() = 50.0
        acc1, acc5 = timm.utils.accuracy(output=validation_resnet_label, target=validation_gt_label, topk=(1, 5))
        accumulate_acc1 += acc1.item()
        accumulate_acc5 += acc5.item()
        accumulate_loss += loss.item()
    # 最后再算一个所有batch的平均准确率
    # print(f"validation result: acc1={accumulate_acc1/accumulate_count}%, acc5={accumulate_acc5/accumulate_count}%")
    return {"acc1": accumulate_acc1 / accumulate_count, "acc5": accumulate_acc5 / accumulate_count,
            "loss": accumulate_loss / accumulate_count}


def train_epoch(model:nn.Module, loss_func:nn.Module, dataloader, optimizer:torch.optim.Optimizer, device:torch.device,
                epoch: int, max_norm: float=0, log_writer=None):
    model.train()

    


def main():
    dataset = build_dataset(is_train=True, dataset_folder='train')
    dataloader = build_dataloader(is_train=True, dataset=dataset)
    dataset_validation = build_dataset(is_train=False, dataset_folder='validation')
    dataloader_validation = build_dataloader(is_train=False, dataset=dataset_validation)
    model = create_model().to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet{resnet_ver}: {n_parameters / 1.e6}M parameters. Model creation success")
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    os.makedirs(log_path, exist_ok=True)
    log_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_path)
    # 训练之前validate一下看看结果
    model.eval()
    validation_result = validation(model=model, dataloader_validation=dataloader_validation, loss_func=loss_func)
    print(f"validation: acc1={validation_result['acc1']},acc5={validation_result['acc5']}, "
          f"loss={validation_result['loss']}")
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}, len(dataloader)={len(dataloader)}", end='')
        loss_list = []

        # 每经过log_step就validate一次
        if epoch % log_step == 0:
            # 模型进入推理模式
            model.eval()
            validation_result = validation(model=model, dataloader_validation=dataloader_validation,
                                           loss_func=loss_func)
            # 写入TensorBoard的SummaryWriter
            if log_writer is not None:
                log_writer.add_scalar(tag="performance/validate_acc1", scalar_value=validation_result['acc1'],
                                      global_step=epoch)
                log_writer.add_scalar(tag="performance/validate_acc5", scalar_value=validation_result['acc5'],
                                      global_step=epoch)
                log_writer.add_scalar(tag="performance/validate_loss", scalar_value=validation_result['loss'],
                                      global_step=epoch)
            # 重回训练模式
            model.train()
            print(f"acc1={validation_result['acc1']},acc5={validation_result['acc5']}, "
                  f"loss={validation_result['loss']}", end='')
            print('\nStart Train...')

        for index_minibatch, sample_minibatch in enumerate(dataloader):
            # 梯度清零
            optimizer.zero_grad()
            # 提取data里面的数据和真实标签
            batch_image, batch_gt_label = sample_minibatch
            # non_blocking参数。这在转移大量数据时尤其有用，因为这允许主机CPU继续向GPU发送数据，而不必等待每次数据传送的完成。
            # 这样在一些情况下可以提高总体的运行速度。然而，这倒不是通常需要担心的问题
            # 因为对于大部分应用，数据传输的成本相对于模型计算的成本来说是微不足道的，除非你的模型非常简单
            # 或者你在使用非常大的数据。因此，大多数情况下，你并不需要设置non_blocking=True。
            batch_image.to(device, non_blocking=True)
            batch_gt_label.to(device, non_blocking=True)
            # 把batch_image传入model
            batch_resnet_label = model(x=batch_image)
            batch_loss = loss_func(batch_resnet_label, batch_gt_label)
            batch_loss.backward()
            optimizer.step()
            loss_list.append(batch_loss)

        loss_avg = torch.mean(torch.FloatTensor(loss_list))


if __name__ == '__main__':
    test_network()
    main()
