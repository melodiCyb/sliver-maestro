import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable



def align(x, y, start_dim=0):
    xd, yd = x.dim(), y.dim()
    if xd > yd: 
        for i in range(xd - yd): 
            y = y.unsqueeze(0) 
    elif yd > xd: 
        for i in range(yd - xd): 
            x = x.unsqueeze(0) 
            
    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if ys[td]==1: 
            ys[td] = xs[td]
        elif xs[td]==1: 
            xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)

def matmul(X,Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i],Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)

def xrecons_grid(batch_size,B,A, T, count=0):
    """
    plots canvas for single time step
    X is x_recons, (batch_size x img_size)
    assumes features = BxA images
    batch is assumed to be a square number
    """
    X = model.generate(batch_size)
    for t in range(T):
        padsize = 1
        padval = .5
        ph = B + 2 * padsize
        pw = A + 2 * padsize
        batch_size = X[t].shape[0]
        N = int(np.sqrt(batch_size))
        X[t] = X[t].reshape((N,N,B,A))
        img = np.ones((N*ph,N*pw))*padval
        
        for i in range(N):
            for j in range(N):
                startr = i * ph + padsize
                endr = startr + B
                startc = j * pw + padsize
                endc = startc + A
                img[startr:endr, startc:endc]=X[t][i, j, :, :]
        img = img[150:180,120:150] # the one at the sixth row and the fifth column
        plt.matshow(img, cmap=plt.cm.gray)
        #plt.show()
        plt.axis('off')
        imgname = '/Users/melodi/submission/baxter-drawing/bankster/src/data/output/images/count_%d_%s_%d.png' % (count,'test', t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
        plt.savefig(imgname)
        print(imgname)

    return img



def save_example_image():
    train_iter = iter(train_loader)
    data, _ = train_iter.next()
    img = data.cpu().numpy().reshape(batch_size, 28, 28)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('/Users/melodi/submission/baxter-drawing/bankster/src/data/output/images/example.png')


def generate():
    x = model.generate(batch_size)
    save_image(x)
    
    
    
 
