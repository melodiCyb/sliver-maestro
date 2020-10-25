import os
import sys

base_path = os.getcwd().split('sliver-maestro')[0]
base_path = os.path.join(base_path, "sliver-maestro")
sys.path.insert(1, base_path)
from src.draw_model import *

config = ConfigParser()
cfg_file = os.path.join(base_path, "src/config.cfg")
config.read(cfg_file)


class Test(DRAW):
    def __init__(self, category, base_path=base_path):
        super().__init__(category, base_path=base_path)
        self.category = category
        self.base_path = base_path
        
    def xrecons(self, index, phase='train'):
        """
        plots canvas for single time step
        X is x_recons, (batch_size x img_size)
        assumes features = BxA images
        batch is assumed to be a square number
        """
        if phase == 'train':
            dataloader = self.dataloaders['train']
        else:
            dataloader = self.dataloaders['test']
     
        batch_iter = (index // self.batch_size) + 1
        for i in range(batch_iter):
            data = dataloader.next_batch(self.batch_size)
            
        # load pre-trained model's weights
        self.load_weights()
        
        # adjust for the model
        img = reshape(data, (data.shape[0], 1, self.A, self.B))
        img = torch.Tensor(img)
        bs = img.size()[0]
        img = Variable(img).view(bs, -1)
        
        # feed forward and generate canvases self.cs
        self.forward(img)
        X = []
       
        for im in self.cs:
            X.append(self.sigmoid(im).cpu().data.numpy())

        #X = self.generate()
        batch_index = index % self.batch_size 
        for t in range(self.T):
            batch_size = X[t].shape[0]
            N = int(np.sqrt(batch_size))
            X[t] = X[t].reshape((N, N, self.B, self.A))
            # initialize for one image
            img = np.ones((1 * self.ph, 1 * self.pw)) * self.padval
           
            i = batch_index // int(np.sqrt(self.batch_size))
            j = batch_index % int(np.sqrt(self.batch_size))
            startr = i * self.ph + self.padsize
            endr = startr + self.B
            startc = j * self.pw + self.padsize
            endc = startc + self.A
            img = X[t][i, j, :, :]
 
            plt.matshow(img, cmap=plt.cm.gray)
            plt.axis('off')
            base_path = os.getcwd().split('sliver-maestro')[0]
            # TODO: refactor naming to match with postprocess.py
            imgname = os.path.join(base_path,
                                   "sliver-maestro",
                                   "src",
                                   "data",
                                   "output",
                                   "images",
                                   self.category,
                                   '%s_index_%d_%d.png' % (self.category, index, t))
            plt.savefig(imgname)
      
        #return img
        
    def xrecons_grid(self, batch=1, phase='train'):
        """
        plots canvas for a chosen batch
        """
        if phase == 'train':
            dataloader = self.dataloaders['train']
        else:
            dataloader = self.dataloaders['test']
            
        for i in range(batch):
            data = dataloader.next_batch(self.batch_size)
            
        # load pre-trained model's weights
        self.load_weights()
        
        # adjust for the model
        img = reshape(data, (data.shape[0], 1, self.A, self.B))
        img = torch.Tensor(img)
        bs = img.size()[0]
        img = Variable(img).view(bs, -1)
        
        # feed forward and generate canvases self.cs
        self.forward(img)
        X = []
       
        for im in self.cs:
            X.append(self.sigmoid(im).cpu().data.numpy())

        #X = self.generate()
        for t in range(self.T):
            batch_size = X[t].shape[0]
            N = int(np.sqrt(batch_size))
            X[t] = X[t].reshape((N, N, self.B, self.A))
            # initialize for the whole batch 
            img = np.ones((N * self.ph, N * self.pw)) * self.padval

            for i in range(N):
                for j in range(N):
                    startr = i * self.ph + self.padsize
                    endr = startr + self.B
                    startc = j * self.pw + self.padsize
                    endc = startc + self.A
                    img[startr:endr, startc:endc] = X[t][i, j, :, :]
          
            plt.matshow(img, cmap=plt.cm.gray)
            plt.axis('off')
            base_path = os.getcwd().split('sliver-maestro')[0]
            imgname = os.path.join(base_path,
                                   "sliver-maestro",
                                   "src",
                                   "data",
                                   "output",
                                   "images",
                                   self.category,
                                   '%s_%s_%d.png' % (self.category, 'batch', t))
            plt.savefig(imgname)

    def load_model(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        saved_path = os.path.join(self.base_path, "src", "save", self.category, '%s.pth' % self.category)
        state = torch.load(saved_path)
        self.load_state_dict(state)

    def load_weights(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        weights_file = os.path.join(self.base_path, "src", "save", self.category,
                                    '%s_weights.tar' % self.category)
        state = torch.load(weights_file, map_location=torch.device('cpu'))
        self.load_state_dict(state)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='sliver-maestro')
    parser.add_argument('-category', '--category')
    parser.add_argument('-idx', '--idx')
    parser.add_argument('-phase', '--phase')
    
    args = parser.parse_args()
    category = args.category or 'cat'
    index = args.idx or 1
    phase = args.phase or 'train'
   
    torch.set_default_tensor_type('torch.FloatTensor')
    test_model = Test(category)
   
    print("generating images...")
    test_model.xrecons(index=int(index), phase=phase)
