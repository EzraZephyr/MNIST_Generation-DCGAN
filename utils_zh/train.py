import csv
import os
import torch
import discriminator
import generator
import dataloader
from torch import nn, optim
from matplotlib import pyplot as plt

def train():
    train_loader = dataloader.data_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 使用CUDA进行训练 如果有的话

    noise_dim = 100
    # 随机噪声向量的维度

    ndf, ngf = 64, 64
    # 判别器和生成器的基础特征图数量

    num_epochs = 100

    netD = discriminator.Discriminator(ndf).to(device)
    netG = generator.Generator(ngf, noise_dim).to(device)
    # 初始化判别器和生成器

    criterion = nn.BCELoss().to(device)
    # 使用二分类交叉熵作为损失函数

    train_csv = '../mnist_train.csv'
    with open(train_csv, 'w', newline='') as f:
        fieldnames = ['Epoch', 'd_loss', 'g_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # 创建用于记录损失的CSV文件

    optimizer_D = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # 使用Adam优化器优化判别器和生成器

    print("Start")

    for epoch in range(num_epochs):
        loss_D, loss_G = 0, 0
        if epoch % 10 == 0:
            save_images(epoch, netG, device)
            # 每10个epoch保存一次生成的样本

        for i, data in enumerate(train_loader, 0):
            netD.zero_grad()
            real_image = data[0].to(device)
            # 获取真实的图像

            batch_size = real_image.size(0)
            # 获取当前批次的大小

            labels = torch.ones(batch_size, device=device) * 0.9
            # 真实标签使用标签平滑用来减少判别器对真实样本的绝对自信

            output = netD(real_image)
            loss_d_real = criterion(output, labels)
            loss_d_real.backward()
            # 计算真实样本的损失并反向传播

            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_image = netG(noise)
            # 生成随机噪声并生成假样本

            labels_fake = torch.zeros(batch_size, device=device) + 0.1
            # 假标签同样也使用标签平滑

            output = netD(fake_image.detach())
            loss_d_fake = criterion(output, labels_fake)
            loss_d_fake.backward()
            # 计算假样本的损失并反向传播

            optimizer_D.step()
            # 更新判别器的参数

            netG.zero_grad()
            labels.fill_(1.0)
            # 生成器希望判别器认为假样本为真 所以标签需要使用1

            output = netD(fake_image)
            loss_g = criterion(output, labels)
            loss_g.backward()
            optimizer_G.step()
            # 计算生成器的损失并反向传播 更新参数

            loss_D += (loss_d_real.item() + loss_d_fake.item())
            loss_G += loss_g.item()

        loss_mean_d = loss_D / len(train_loader)
        loss_mean_g = loss_G / len(train_loader)
        # 计算平均损失

        print(f'Epoch: {epoch+1}, Loss_D: {loss_mean_d:.4f}, Loss_G: {loss_mean_g:.4f}')

        with open(train_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch+1, 'd_loss': loss_mean_d, 'g_loss': loss_mean_g})
            # 将当前epoch的损失写入文件

    torch.save(netD.state_dict(), '../model/netD.pth')
    torch.save(netG.state_dict(), '../model/netG.pth')

    return netG

def save_images(epoch, netG, device):
    noise = torch.randn(64, 100, 1, 1, device=device)
    with torch.no_grad():
        fake_images = netG(noise)
    fake_images = fake_images.cpu().detach().numpy()
    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            img = fake_images[i * 8 + j, 0, :, :]
            ax[i, j].imshow(img, cmap='gray')
            ax[i, j].axis('off')
    if not os.path.exists('../images'):
        os.makedirs('../images')
    plt.savefig(f'../images/epoch_{epoch}.png', bbox_inches='tight')
    plt.close()
    # 生成并保存8x8网格的图像

if __name__ == '__main__':
    netG = train()
    save_images(100, netG, device='cuda')
    # 训练结束后生成并保存最终样本
