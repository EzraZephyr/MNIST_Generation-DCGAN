from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loader():
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # 将每张图片缩放为28x28转换为张量tensor并进行标准化

    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    # 下载MNIST数据集并应用转换、缩放和标准化

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # 按批次加载数据 每个批次包含128张图片并打乱数据顺序

    return train_loader
