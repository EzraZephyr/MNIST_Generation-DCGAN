import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngf, noise_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(noise_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ELU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ELU(True),

            nn.ConvTranspose2d(ngf, 1, 5, 1, 0, bias=False),
            nn.Tanh()
        )
        # 将输入的随机噪声映射为一张图片 使用更适合深层网络的ELU激活函数
        # 最后一层通过Tanh将输出归一化到-1到1的区间让生成的图像与输入数据的分布一致


    def forward(self, input):
        output = self.main(input)
        output = output[:, :, :28, :28]
        # 裁剪输出到28x28的大小
        return output
