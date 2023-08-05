# GAN model experiment 2023-08-02
# GAN，一个天才的设计，对抗式生成网络
# 两个网络，一个生成器Generator，一个判别器Discriminator
# 两个网络相互对抗，最终达到一个平衡，生成器生成的图片越来越真实，判别器判别的越来越准确
# Genius

import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import os

# 1 为channel，28为height，28为width
image_size = (1, 28, 28)
image_size_int = torch.prod(torch.tensor(image_size), dtype=torch.int32).item()
# 10个数字需要区分
mnist_discrimination_number = 10
num_epoch = 1000
batch_size = 40
latent_dim = 100
epsilon_gan = 0.01
device = None

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")


class Generator(nn.Module):
    # TODO: in_dim的默认维度应该是什么呢?还得再好好想想
    def __init__(self, in_dim=latent_dim, *args, **kwargs):
        # 这里的args是直接传的参数func(1)，kwargs是字典参数func(name="some guy")
        super(Generator, self).__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            # torch.prod计算image_size各个维度的乘积，.item()转换为int
            # 这一步其实就是把1024隐空间的向量转换为image，此为：Generator哒！
            nn.Linear(1024, image_size_int),
            # 将输出映射到-1:1中，图像灰度图
            nn.Tanh()
        )

    def forward(self, z_g):
        # z是一个[batch_size, latent_dim]
        # 这里的latent_dim是什么呢？是隐空间的维度，也就是说，这里的z是一个隐空间的向量
        # GAN的目标就是把randn()生成的随机向量z，映射到真实图片的分布上，所以latent_dim可以是任意维度，这是一个超参数
        batch_size_g = z_g.shape[0]
        output = self.model(z_g)
        assert (isinstance(output, torch.Tensor))
        # 这里的*image_size是解包，将image_size中的三个值传入reshape函数，把元组解包为三个参数，最终是(batch_size, 1, 28, 28)
        image_batches = output.reshape((batch_size_g, *image_size))

        return image_batches


class Discriminator(nn.Module):
    def __init__(self, in_dim=image_size_int, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            # 二分类，输出0或1，也就是Discriminator在看到一张图片后，判断这张图片是真的还是假的
            nn.Linear(256, 1),
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
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                             # TODO: 应该是多少最好呢？
                                             mean=(0.5),
                                             std=(0.5)
                                         )
                                     ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

train_option = input("(L)oad previous model or (T)rain a new model? or (G)enerate images?")
start_epoch = 0
if train_option == "L" or train_option == "l":
    generator.load_state_dict(torch.load("./GAN_data/GAN_results_ckpt/generator.pth"))
    discriminator.load_state_dict(torch.load("./GAN_data/GAN_results_ckpt/discriminator.pth"))
    start_epoch = input("Load previous model successfully! The previous epoch number: ")
    start_epoch = int(start_epoch)
elif train_option == "T" or train_option == "t":
    print("Start training a new model!")
elif train_option == "G" or train_option == "g":
    generator.load_state_dict(torch.load("./GAN_data/GAN_results_ckpt/generator.pth"))
    generate_img_option = input("Generate Image number: ")
    generate_img_option = int(generate_img_option)
    z_gen = torch.randn(generate_img_option, latent_dim).to(device)
    generated_image = generator(z_gen)
    torchvision.utils.save_image(generated_image, f"./GAN_data/GAN_generate/image_gen_{generate_img_option}.png")
    exit(0)
else:
    print("Invalid input!")
    exit(0)

print("Start training... Create file stop_gan.txt to stop it.")

generator_optim = torch.optim.Adam(params=generator.parameters(), lr=0.0002)
discriminator_optim = torch.optim.Adam(params=discriminator.parameters(), lr=0.0002)

loss_func = nn.BCELoss()

stop_count = 5
stop_counter = 0

# 开始训练
for index_epoch in range(start_epoch + 1, num_epoch):
    loss_g_list = []
    loss_d_list = []
    for index_minibatch, sample_minibatch in enumerate(dataloader):
        # generator和discriminator的梯度置零
        generator_optim.zero_grad()
        discriminator_optim.zero_grad()

        # 获取mini_batch, 随机生成latent_dim大小的正态分布，准备让generator映射
        gt_image, label = sample_minibatch
        gt_image = gt_image.to(device)
        z = torch.randn(batch_size, latent_dim).to(device)
        # 生成32张照片
        gen_images = generator(z)
        # 判别：一眼丁真？
        # dis_prob = discriminator(gen_images)

        # discriminator梯度下降
        target_dis_gt = torch.ones(batch_size, 1).to(device)
        # ground-truth
        loss_d_gt = loss_func(discriminator(gt_image), target_dis_gt)
        # generator-adversarial
        # 在训练生成器时，我们需要通过生成器生成一批样本，并将这些样本输入到判别器中以获取判别器的输出。
        # 然后我们要计算生成器的损失并进行反向传播更新参数。注意，我们只想更新生成器的参数，而不想影响判别器。
        # 由于生成器的输出gen_images是通过生成器进行前向传播得到的，因此它与生成器参数存在依赖关系。
        # 如果我们直接在计算生成器损失时使用gen_images，梯度将通过gen_images反向传播到生成器的参数，从而影响生成器的参数更新。
        # 为了避免这种情况，我们使用.detach()函数将gen_images从计算图中分离出来，这样就可以阻止梯度通过gen_images反向传播到生成器的参数。
        # 这样，只会更新判别器的参数，生成器的参数将保持不变。
        # 所以在这种情况下，我们需要使用gen_images.detach()将生成器的输出分离出来，以便在生成器的反向传播中阻止梯度传播。
        target_dis_GA = torch.zeros(batch_size, 1).to(device)
        # 这一行尤为关键！！之所以Generator要在Discriminator的后面训练，是因为这一部有discriminator(gen_images.detach())
        # 只有这样，Generator才能在第一步的时候，由于discriminator已经被ground-truth训练过了，所以有一定基础判断能力
        # 而不至于两个Discriminator，generator在互相摆烂，就这一部，非常关键
        # Discriminator永远要在Generator前面训练
        loss_d_GA = loss_func(discriminator(gen_images.detach()), target_dis_GA)
        loss_d = 0.5 * (loss_d_gt + loss_d_GA)
        loss_d.backward()
        discriminator_optim.step()

        # generator梯度下降
        # 之所以是全部1(ones)，是因为我们希望generator生成的图片被判别为真，或者说这是它应该努力的方向
        target_gen = torch.ones(batch_size, 1).to(device)
        loss_g = loss_func(discriminator(gen_images), target_gen)
        loss_g.backward()
        generator_optim.step()

        loss_g_list.append(loss_g)
        loss_d_list.append(loss_d)

    loss_g_avg = torch.mean(torch.FloatTensor(loss_g_list))
    loss_d_avg = torch.mean(torch.FloatTensor(loss_d_list))
    gen_image_final = generator(torch.randn(batch_size, latent_dim).to(device))
    print(f"epoch [{index_epoch}/{num_epoch}] gen_loss:{loss_g_avg}, dis_loss:{loss_d_avg}")
    torchvision.utils.save_image(gen_image_final, f"./GAN_data/GAN_results/image_gen_{index_epoch}.png")
    if os.path.exists("stop_gan.txt"):
        abort_option = input("stop_gan.txt detected, (S)ave model and exit, (C)ontinue training, (SC) Save and "
                             "Continue training")
        if abort_option == "S" or abort_option == "s" or abort_option == "SC" or abort_option == "sc":
            torch.save(generator.state_dict(), "./GAN_data/GAN_results_ckpt/generator.pth")
            torch.save(discriminator.state_dict(), "./GAN_data/GAN_results_ckpt/discriminator.pth")
            if abort_option == "S" or abort_option == "s":
                break
    if abs(loss_d_avg - 0.5) < epsilon_gan:
        stop_counter += 1
        if stop_counter == stop_count:
            print("Multiple epoch trained meet requirement(loss_g approx 0.5), stopping training")
            torch.save(generator.state_dict(), "./GAN_data/GAN_results_ckpt/generator.pth")
            torch.save(discriminator.state_dict(), "./GAN_data/GAN_results_ckpt/discriminator.pth")
            break
    else:
        stop_counter = 0


torch.save(generator.state_dict(), "./GAN_data/GAN_results_ckpt/generator.pth")
torch.save(discriminator.state_dict(), "./GAN_data/GAN_results_ckpt/discriminator.pth")