import os
import numpy as np
from sklearn.model_selection import train_test_split
from os import walk
import matplotlib.pyplot as plt
import numpy as np
import torch

# ------------
# dataset.py
# ------------

class Dataset:

    def __init__(self, data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    def next_batch(self, batch_size, shuffle=False):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            if shuffle:
                np.random.shuffle(idx)
            self._data = self.data[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)
            if shuffle:
                # shuffle indexes
                np.random.shuffle(idx0)  
            self._data = self.data[idx0]

            start = 0
            # avoid the case where the #sample != integar times of batch_size
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            print('start, end:{}, {}'.format(start, end))
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            print('start, end:{}, {}'.format(start, end))
            return self._data[start:end]


# Place all the npy quickdraw files in filepath
# MC: Do we need this function?
def split_data_multiple_files(filepath, category=False, shuffle=False):
    txt_name_list = []
    if category:
        filepath = os.path.join(filepath, category + '\\') 
        for (dirpath, dirnames, filenames) in walk(filepath):
            for filename in filenames:
                if filename.endswith('.npy'):
                    txt_name_list.append(filename)
                    break
    else:
        for (dirpath, dirnames, filenames) in walk(filepath):
            if filenames != '.DS_Store':
                txt_name_list.extend(filenames)
                break

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    xtotal = []
    ytotal = []
    # Setting value to be 80000 for the final dataset
    slice_train = int(80000 / len(txt_name_list))
    i = 0
    seed = np.random.randint(1, 10e6)

    # Creates test/train split with quickdraw data
    for txt_name in txt_name_list:
        txt_path = filepath + txt_name
        xx = np.load(txt_path)
        try:
            xx = xx.astype('float32') / 255.  # scale images to binary
        except AttributeError:
            pass
        try:
            y = [i] * len(xx)
            if shuffle:
                np.random.seed(seed)
                np.random.shuffle(xx)
                np.random.seed(seed)
                np.random.shuffle(y)
            xx = xx[:slice_train]
            y = y[:slice_train]
            if i != 0:
                xtotal = np.concatenate((xx, xtotal), axis=0)
                ytotal = np.concatenate((y, ytotal), axis=0)
            else:
                xtotal = xx
                ytotal = y
            i += 1
            if shuffle:
                x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.2, random_state=42)
            else:
                x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.2, shuffle=False)
        except TypeError:
            pass
    return x_train, x_test, y_train, y_test

# Place all the npy quickdraw files in filepath
def split_data(filepath, category, shuffle=False):
    # Creates test/train split with quickdraw data
    filepath = os.path.join(filepath, category + '\\' + '%s.npy' % category) 
    xx = np.load(filepath)
    y = [0] * len(xx)
    try:
        xx = xx.astype('float32') / 255.
    except AttributeError:
        pass
    if shuffle:
        x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=0.2, random_state=42)
    else:
        x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=0.2, shuffle=False)
    return x_train, x_test, y_train, y_test

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

# ------------------
# create_folders.py
# ------------------

def create_folders(root_path, categories, category):

    data_path = os.path.join(root_path + "src" + "\\" + "data")
    save_path = os.path.join(root_path + "src" + "\\" + "save")
    input_path = os.path.join(data_path + "\\" + "input")
    output_path = os.path.join(data_path + "\\" + "output")
    images_path = os.path.join(output_path + "\\" + "images")
    positions_path = os.path.join(output_path + "\\" + "positions")
    raw_path = os.path.join(data_path + "\\" + "raw")
    category_save_path = os.path.join(save_path + "\\" + categories[category])
    category_input_path = os.path.join(data_path + "\\" + "input" + "\\" + categories[category])
    category_raw_path = os.path.join(data_path + "\\" + "raw" + "\\" + categories[category])
    category_images_path = os.path.join(images_path + "\\" + categories[category])
    base_input = os.path.join(category_input_path + "\\" + categories[category])
    base_raw = os.path.join(category_raw_path + "\\" + categories[category])
    
    paths_dict = {'data_path':data_path,
                  'save_path':save_path,
                  'input_path':input_path,
                  'output_path':output_path,
                  'images_path':images_path,
                  'positions_path':positions_path,
                  'raw_path':raw_path,
                  'category_save_path':category_save_path,
                  'category_input_path':category_input_path,
                  'category_raw_path':category_raw_path,
                  'category_images_path':category_images_path,
                  'base_input':base_input,
                  'base_raw':base_raw
                 }
    
    src_list = []
    dst_list = []

    src_file = "gs://quickdraw_dataset/full/numpy_bitmap/" + categories[category] + ".npy"
    src_list.append(src_file)

    src_file = "gs://quickdraw_dataset/full/raw/" + categories[category] + ".ndjson"
    src_list.append(src_file)

    dst_file_input = os.path.join(category_input_path + "\\" + categories[category])
    dst_file_raw = os.path.join(category_raw_path + "\\" + categories[category])

    dst_list.append(dst_file_input + ".npy")
    dst_list.append(dst_file_raw + ".ndjson")
    
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        if os.path.exists(data_path):
            print("{} folder is created...".format(data_path))
    else:
         print("{} folder exists...".format(data_path))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        if os.path.exists(save_path):
            print("{} folder is created...".format(save_path))
    else:
         print("{} folder exists...".format(save_path))
            
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        if os.path.exists(input_path):
            print("{} folder is created...".format(input_path))        
    else:
         print("{} folder exists...".format(input_path))
            
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        if os.path.exists(output_path):
            print("{} folder is created...".format(output_path))
    else:
         print("{} folder exists...".format(output_path))

    if not os.path.exists(images_path):
        os.mkdir(images_path)
        if os.path.exists(images_path):
            print("{} folder is created...".format(images_path))
    else:
         print("{} folder exists...".format(images_path))
            
    if not os.path.exists(positions_path):
        os.mkdir(positions_path)
        if os.path.exists(positions_path):
            print("{} folder is created...".format(positions_path))
    else:
         print("{} folder exists...".format(positions_path))
            
    if not os.path.exists(raw_path):
        os.mkdir(raw_path)
        if os.path.exists(raw_path):
            print("{} folder is created...".format(raw_path))
    else:
         print("{} folder exists...".format(raw_path))

    if not os.path.exists(category_save_path):
        os.mkdir(category_save_path)
        if os.path.exists(category_save_path):
            print("{} folder is created...".format(category_save_path))
    else:
         print("{} folder exists...".format(category_save_path))         
            
    if not os.path.exists(category_input_path):
        os.mkdir(category_input_path)
        if os.path.exists(category_input_path):
            print("{} folder is created...".format(category_input_path))
    else:
         print("{} folder exists...".format(category_input_path))            
    
    if not os.path.exists(category_raw_path):
        os.mkdir(category_raw_path)
        if os.path.exists(category_raw_path):
            print("{} folder is created...".format(category_raw_path))
    else:
         print("{} folder exists...".format(category_raw_path))   

    if not os.path.exists(category_images_path):
        os.mkdir(category_images_path)
        if os.path.exists(category_images_path):
            print("{} folder is created...".format(category_images_path))
    else:
         print("{} folder exists...".format(category_images_path))   
            
    return src_list, dst_list, paths_dict

# ---------------
# model_utils.py
# ---------------

def matmul(X,Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i],Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)

def xrecons_grid(batch_size, B, A, T, category, model, img_loc, count=0):
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
        img = img[img_loc['startr']:img_loc['endr'],img_loc['startc']:img_loc['endc']] # the one at the sixth row and the fifth column
        plt.matshow(img, cmap=plt.cm.gray)
        plt.axis('off')
        root_path = os.getcwd().split('notebook')[0]
        imgname = os.path.join(root_path,
                               "src",
                               "data",
                               "output",
                               "images",
                               category,
                               '%s_%d_%s_%d.png' % (category, count,'test', t))
        plt.savefig(imgname)
        print(imgname)

    return img

