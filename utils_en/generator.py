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
        # Map the input random noise into an image,
        # using ELU activation functions that are better suited for deeper networks.
        # The last layer uses Tanh to normalize the output to the range of -1 to 1,
        # aligning the generated image with the distribution of the input data.

    def forward(self, input):
        output = self.main(input)
        output = output[:, :, :28, :28]
        # Crop the output to a size of 28x28
        return output
