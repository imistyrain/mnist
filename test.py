#https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
import cv2
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import ConvNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

embeddings = None

def draw_scatter(embeddings, targets, embedding_size=2):
    cmap = cm.get_cmap('tab10')
    if embedding_size == 2:
        fig, ax = plt.subplots(figsize=(8,8))
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    plt.title('logit distribution')
    num_categories = 10
    for lab in range(num_categories):
        indices = targets==lab
        pi = embeddings[indices]
        if embedding_size==2:
            ax.scatter(pi[:,0],pi[:,1], color=cmap(lab), label = lab, alpha=0.5)
        else:
            ax.scatter(pi[:,0],pi[:,1],pi[:,2],color=cmap(lab), label = lab, alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig("output/dist.png")
    #plt.show()

def test(embedding_size=2):
    dataset = datasets.MNIST(root = 'data/', train = False, transform = transforms.ToTensor(), download = True)
    data_loader = DataLoader(dataset = dataset, batch_size= 10000, shuffle = False)
    model = ConvNet(embedding_size)
    checkpoint = torch.load("models/"+str(embedding_size)+"d/best.pth")
    checkpoint = {k.replace('module.',''):v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    rows = 12
    index = 0
    correct = 0
    total = 0
    
    def hook(module, input, output):
        global out
        out = output
        return None
    model.fc2.register_forward_hook(hook)
    fc3 = model.fc3.weight.data
    print(fc3[:,0])
    embeddings = np.zeros(shape=(0,embedding_size))
    targets = np.zeros(shape=(0))
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for images, labels in pbar:
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            acc = 100 * correct / total
            pbar.set_description("Acc{:.4f}".format(acc))

            embeddings = np.concatenate([embeddings, out.detach().cpu().numpy()],axis=0)
            targets = np.concatenate((targets,labels.numpy().ravel()))
            # img = torchvision.utils.make_grid(images, nrow = rows)
            # img = img.numpy().transpose(1, 2, 0)
            # for k, pred in enumerate(preds):
            #     image = images[k].cpu().numpy().transpose(1,2,0)
            #     # cv2.imwrite("output/test.jpg",image*255)
            #     i = k // rows
            #     j = k % rows
            #     x = j*32
            #     y = i*32
            #     pred = pred.item()
            #     label = labels[k].item()
            #     if label != pred:
            #         dst_path = "error/"+str(index)+"_"+str(label)+"_"+str(pred)+".jpg"
            #         cv2.imwrite(dst_path, image*255)
            #     #info = str(pred.item())
            #     #print(info)
            #     #cv2.putText(img,info,(x,y),1,1,(0,255,0))
            #     index += 1
            # cv2.imshow('img', img)
            # cv2.waitKey()
            # cv2.imwrite("output/"+str(index)+".jpg", img*255)

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))
        draw_scatter(embeddings, targets, embedding_size)

if __name__=="__main__":
    test()