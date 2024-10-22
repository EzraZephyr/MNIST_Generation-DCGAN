from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loader():
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # Resize each image to 28x28, convert it to a tensor, and normalize it

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # Download the MNIST dataset and apply transformations, resizing, and normalization

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # Load data in batches, with each batch containing 128 images and shuffle the data

    return train_loader
