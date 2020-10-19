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

    def xrecons_grid(self, index, img):
        """
        plots canvas for single time step
        X is x_recons, (batch_size x img_size)
        assumes features = BxA images
        batch is assumed to be a square number
        """
        
        self.load_model()
        
        # adjust for the model
        img = reshape(img, (img.shape[0], 1, self.A, self.B))
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
            padsize = 1
            padval = .5
            ph = self.B + 2 * padsize
            pw = self.A + 2 * padsize
            batch_size = X[t].shape[0]
            N = int(np.sqrt(batch_size))
            X[t] = X[t].reshape((N, N, self.B, self.A))
            img = np.ones((N * ph, N * pw)) * padval

            for i in range(N):
                for j in range(N):
                    startr = i * ph + padsize
                    endr = startr + self.B
                    startc = j * pw + padsize
                    endc = startc + self.A
                    img[startr:endr, startc:endc] = X[t][i, j, :, :]
            img_loc = self.get_image_location(index) 
            img = img[img_loc['startr']:img_loc['endr'], img_loc['startc']:img_loc['endc']]
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
                                   '%s_%s_%d.png' % (self.category, 'test', t))
            plt.savefig(imgname)
            print(imgname)

        return img

    def load_model(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        saved_path = os.path.join(self.base_path, "src", "save", self.category, '%s.pth' % self.category)
        state = torch.load(saved_path)
        self.load_state_dict(state)

    def load_weights(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        weights_file = os.path.join(self.base_path, "sliver-maestro", "src", "save", self.category,
                                    '%s_weights.tar' % self.category)
        state_dict = torch.load(weights_file, map_location=torch.device('cpu'))
        model = self.model.load_state_dict(state_dict)

        return model
    
    def get_image_location(self, index):

        startr = (index // 8) * 30
        endr = ((index // 8) * 30) + 30

        startc = (index % 8) * 30
        endc = ((index % 8) * 30) + 30

        return {'startr':startr, 'endr':endr, 'startc':startc, 'endc':endc}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='sliver-maestro')
    parser.add_argument('-category', '--category')

    args = parser.parse_args()
    category = args.category
    if not category:
        category = 'cat'

    torch.set_default_tensor_type('torch.FloatTensor')
    test_model = Test(category)
    ## TODO: add dataloaders['test'] option together with the start index info
    dataloader = test_model.dataloaders['train']
    data = dataloader.next_batch(test_model.batch_size)
    
    
    img_loc = {'startr': 0, 'endr': 30, 'startc': 0, 'endc': 30}
    print("generating images...")
    test_model.xrecons_grid(img_loc, data)
