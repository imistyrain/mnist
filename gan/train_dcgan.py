#https://blog.csdn.net/jining11/article/details/89644051
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
            
def show_weights_hist(data):
    plt.hist(data, bins=100, normed=1, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("weights")
    plt.ylabel("frequency")
    plt.title("D weights")
    plt.show()

class DCGenerator(nn.Module):
    def __init__(self, image_size=32, latent_dim=64, output_channel=1):
        super(DCGenerator, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.output_channel = output_channel
        
        self.init_size = image_size // 8
        # 相当于除了三次2，因此在反卷积中要乘三次二变回来
        
        # fc: Linear -> BN -> ReLU
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size ** 2),
            nn.BatchNorm1d(512 * self.init_size ** 2),
            nn.ReLU(inplace=True)
        )
        
        # deconv: ConvTranspose2d(4, 2, 1) -> BN -> ReLU -> 
        #         ConvTranspose2d(4, 2, 1) -> BN -> ReLU -> 
        #         ConvTranspose2d(4, 2, 1) -> Tanh
        # 根据o = s(i-1)-2p+k，o = s(i)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, output_channel, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.deconv(out)
        return img

class DCDiscriminator(nn.Module):
    def __init__(self, image_size=32, input_channel=1, sigmoid=True):
        super(DCDiscriminator, self).__init__()
        self.image_size = image_size
        self.input_channel = input_channel
        self.fc_size = image_size // 8
        
        # conv: Conv2d(3,2,1) -> LeakyReLU 
        #       Conv2d(3,2,1) -> BN -> LeakyReLU 
        #       Conv2d(3,2,1) -> BN -> LeakyReLU 
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # fc: Linear -> Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(512 * self.fc_size * self.fc_size, 1),
        )
        if sigmoid:
            self.fc.add_module('sigmoid', nn.Sigmoid())
        initialize_weights(self)
        
        

    def forward(self, img):
        out = self.conv(img)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out

def load_mnist_data():
    """
    load mnist(0,1,2) dataset 
    """
    
    transform = torchvision.transforms.Compose([
        # transform to 1-channel gray image since we reading image in RGB mode
        transforms.Grayscale(1),
        # resize image from 28 * 28 to 32 * 32
        transforms.Resize(32),
        transforms.ToTensor(),
        # normalize with mean=0.5 std=0.5
        transforms.Normalize(mean=(0.5, ),  std=(0.5, ))
        ])
    
    #train_dataset = torchvision.datasets.ImageFolder(root='data', transform=transform)
    train_dataset = torchvision.datasets.MNIST("data", train=True, download=True, transform=transform)
    return train_dataset

def load_furniture_data():
    """
    load furniture dataset 
    """
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        # normalize with mean=0.5 std=0.5
        transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                             std=(0.5, 0.5, 0.5))
        ])
    train_dataset = torchvision.datasets.ImageFolder(root='./data/household_furniture', transform=transform)
    return train_dataset

def denorm(x):
    # 这个函数的功能是把正规化之后的数据还原到原始数据
    # denormalize
    # normalize:y = (x-0.5)/0.5, y/2 = x-1/2, y+1 = 2x, x = (y+1)/2
    out = (x + 1) / 2
    # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内
    return out.clamp(0, 1)

def show(img):
    # 显示图片
    npimg = img.numpy()
    # 最近插值法(Nearest Neighbor Interpolation)：这是一种最简单的插值算法，输出像素的值为输入图像离映射点最近的像素值
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def show_gt():
    # show mnist real data
    train_dataset = load_mnist_data()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    show(torchvision.utils.make_grid(denorm(next(iter(trainloader))[0]), nrow=10))

def train(trainloader, G, D, G_optimizer, D_optimizer, loss_func, device, z_dim):
    """
    train a GAN with model G and D in one epoch
    Args:
        trainloader: data loader to train
        G: model Generator
        D: model Discriminator
        G_optimizer: optimizer of G(etc. Adam, SGD)
        D_optimizer: optimizer of D(etc. Adam, SGD)
        loss_func: loss function to train G and D. For example, Binary Cross Entropy(BCE) loss function
        device: cpu or cuda device
        z_dim: the dimension of random noise z
    """
    # set train mode
    D.train()
    G.train()
    
    D_total_loss = 0
    G_total_loss = 0
    
    pbar = tqdm(trainloader)
    for i, (x, _) in enumerate(pbar):
        # real label and fake label
        y_real = torch.ones(x.size(0), 1).to(device)
        y_fake = torch.zeros(x.size(0), 1).to(device)
        # batch_size个真实数据
        x = x.to(device)
        # batch_size个z_dim维的随机噪声
        z = torch.rand(x.size(0), z_dim).to(device)

        # update D network
        # D optimizer zero grads
        D_optimizer.zero_grad()
        
        # D real loss from real images
        d_real = D(x)
        d_real_loss = loss_func(d_real, y_real)
        
        # D fake loss from fake images generated by G
        g_z = G(z)
        d_fake = D(g_z)
        d_fake_loss = loss_func(d_fake, y_fake)
        
        # D backward and step
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        D_optimizer.step()

        # update G network
        # G optimizer zero grads
        G_optimizer.zero_grad()
        
        # G loss
        g_z = G(z)
        d_fake = D(g_z)
        g_loss = loss_func(d_fake, y_real)
        
        # G backward and step
        g_loss.backward()
        G_optimizer.step()
        
        D_total_loss += d_loss.item()
        G_total_loss += g_loss.item()
        msg = 'D loss: {:.4f}, G loss: {:.4f}'.format(d_loss, g_loss)
        pbar.set_description(msg)
    
    return D_total_loss / len(trainloader), G_total_loss / len(trainloader)

def visualize_results(G, device, z_dim, epoch=0, result_size=25):
    G.eval()
    z = torch.rand(result_size, z_dim).to(device)
    g_z = G(z)
    #show(torchvision.utils.make_grid(denorm(g_z.detach().cpu()), nrow=5))
    save_image(g_z.data, "images/dcgan/%d.png" % epoch, nrow=5, normalize=True) # 保存一个batchsize中的25张

def run_gan(trainloader, G, D, G_optimizer, D_optimizer, loss_func, n_epochs, device, latent_dim):
    d_loss_hist = []
    g_loss_hist = []

    for epoch in range(n_epochs):
        d_loss, g_loss = train(trainloader, G, D, G_optimizer, D_optimizer, loss_func, device, 
                               z_dim=latent_dim)
        print('Epoch {}: Train D loss: {:.4f}, G loss: {:.4f}'.format(epoch, d_loss, g_loss))

        d_loss_hist.append(d_loss)
        g_loss_hist.append(g_loss)

        #if epoch == 0 or (epoch + 1) % 10 == 0:
        visualize_results(G, device, latent_dim, epoch)
        torch.save(G,'models/dcgan/g%d.pth' % epoch)
        torch.save(D,'models/dcgan/d%d.pth' % epoch)
    
    return d_loss_hist, g_loss_hist

def loss_plot(d_loss_hist, g_loss_hist):
    x = range(len(d_loss_hist))

    plt.plot(x, d_loss_hist, label='D_loss')
    plt.plot(x, g_loss_hist, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def main():
    # hyper params
    # z dim
    latent_dim = 100

    # image size and channel
    image_size=32
    image_channel=1

    # Adam lr and betas
    learning_rate = 0.0002
    betas = (0.5, 0.999)

    # epochs and batch size
    n_epochs = 100
    batch_size = 32

    # device : cpu or cuda:0/1/2/3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # mnist dataset and dataloader
    train_dataset = load_mnist_data()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # use BCELoss as loss function
    bceloss = nn.BCELoss().to(device)

    # G and D model
    G = DCGenerator(image_size=image_size, latent_dim=latent_dim, output_channel=image_channel).to(device)
    D = DCDiscriminator(image_size=image_size, input_channel=image_channel).to(device)

    # G and D optimizer, use Adam or SGD
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=betas)

    d_loss_hist, g_loss_hist = run_gan(trainloader, G, D, G_optimizer, D_optimizer, bceloss, n_epochs, device, latent_dim)  
    loss_plot(d_loss_hist, g_loss_hist)

if __name__=="__main__":
    main()