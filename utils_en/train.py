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
    # Use CUDA for training if available

    noise_dim = 100
    # Dimension of the random noise vector

    ndf, ngf = 64, 64
    # Base feature map counts for the discriminator and generator

    num_epochs = 100

    netD = discriminator.Discriminator(ndf).to(device)
    netG = generator.Generator(ngf, noise_dim).to(device)
    # Initialize the discriminator and generator

    criterion = nn.BCELoss().to(device)
    # Use binary cross-entropy as the loss function

    train_csv = '../mnist_train.csv'
    with open(train_csv, 'w', newline='') as f:
        fieldnames = ['Epoch', 'd_loss', 'g_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Create a CSV file to record the losses

    optimizer_D = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # Use the Adam optimizer to optimize the discriminator and generator

    print("Start")

    for epoch in range(num_epochs):
        loss_D, loss_G = 0, 0
        if epoch % 10 == 0:
            save_images(epoch, netG, device)
            # Save generated samples every 10 epochs

        for i, data in enumerate(train_loader, 0):
            netD.zero_grad()
            real_image = data[0].to(device)
            # Get real images

            batch_size = real_image.size(0)
            # Get the size of the current batch

            labels = torch.ones(batch_size, device=device) * 0.9
            # Use label smoothing for real labels to reduce the discriminator's absolute confidence in real samples

            output = netD(real_image)
            loss_d_real = criterion(output, labels)
            loss_d_real.backward()
            # Compute the loss for real samples and backpropagate

            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_image = netG(noise)
            # Generate random noise and create fake samples

            labels_fake = torch.zeros(batch_size, device=device) + 0.1
            # Also use label smoothing for fake labels

            output = netD(fake_image.detach())
            loss_d_fake = criterion(output, labels_fake)
            loss_d_fake.backward()
            # Compute the loss for fake samples and backpropagate

            optimizer_D.step()
            # Update the discriminator's parameters

            netG.zero_grad()
            labels.fill_(1.0)
            # The generator aims for the discriminator to classify fake samples as real, so the labels are set to 1

            output = netD(fake_image)
            loss_g = criterion(output, labels)
            loss_g.backward()
            optimizer_G.step()
            # Compute the generator's loss, backpropagate, and update parameters

            loss_D += (loss_d_real.item() + loss_d_fake.item())
            loss_G += loss_g.item()

        loss_mean_d = loss_D / len(train_loader)
        loss_mean_g = loss_G / len(train_loader)
        # Calculate the average losses

        print(f'Epoch: {epoch+1}, Loss_D: {loss_mean_d:.4f}, Loss_G: {loss_mean_g:.4f}')

        with open(train_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch+1, 'd_loss': loss_mean_d, 'g_loss': loss_mean_g})
            # Write the loss of the current epoch to the file

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
    # Generate and save a 8x8 grid of images

if __name__ == '__main__':
    netG = train()
    save_images(100, netG, device='cuda')
    # Generate and save final samples after training
