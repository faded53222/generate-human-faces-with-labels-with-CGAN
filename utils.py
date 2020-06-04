import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

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
    

def show(img, figsize=None):
    npimg = img.numpy()
    if figsize is not None:
        plt.figure(figsize=(figsize,figsize))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.pause(0.05)
    

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            
def show_weights_hist(data):
    plt.hist(data, bins=100, normed=1, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("weights")
    plt.ylabel("frequency")
    plt.title("D weights")
    plt.show()

def weights_init_normal(m):
    '''
        initial the weight in normal distribution
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    
