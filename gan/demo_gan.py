import os
import torch
import numpy as np
from torchvision.utils import save_image
from train_gan import Generator

def get_latest_model(dir='models'):
    files = os.listdir(dir)
    latest = 1
    checkpoint_path = ''
    ret = ''
    for file in files:
        checkpoint_path = dir+'/'+file
        if os.path.isfile(checkpoint_path) and file.startswith('g'):
            epoch = int(file[1:-4])
            if epoch > latest:
                latest = epoch
                ret = checkpoint_path
    return ret

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latest = get_latest_model()
print(latest)

g = torch.load(latest) #导入生成器Generator模型
g = g.to(device)
g.eval()
z = torch.Tensor(np.random.normal(0, 1, (64, 100))).to(device)
gen_imgs =g(z) #生产图片
save_image(gen_imgs.data[:25], "images/gan.png" , nrow=5, normalize=True)