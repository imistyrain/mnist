import cv2
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def show_gt():
    dataset = datasets.MNIST(root = 'data/', train = True, transform = transforms.ToTensor(), download = True)
    data_loader = DataLoader(dataset = dataset, batch_size = 256, shuffle = True)
    rows = 16
    index = 0
    for images, labels in tqdm(data_loader):
        img = torchvision.utils.make_grid(images, nrow = rows)
        img = img.numpy().transpose(1, 2, 0)
        #cv2.imshow('img', img)
        #cv2.waitKey()
        cv2.imwrite("output/gt/"+str(index)+".jpg", img*255)
        index += 1

if __name__=="__main__":
    show_gt()