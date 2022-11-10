import torch
import torchvision
import torchvision.transforms as transforms

def eval(model, device):
    test_dataset = torchvision.datasets.MNIST( root='./data', train=False, transform=transforms.ToTensor(),download=True)   
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            acc = 100 * correct / total
        return acc