import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(ndf * 4, 1,4,1,0, bias=False),
            nn.Sigmoid()
            # 判别器通过多层卷积和激活函数逐步提取输入图像的特征 最终通过sigmoid输出一个概率值 该值表示输入图像为真实图像的可能性
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
        # 进行前向传播并移除多余的维度
