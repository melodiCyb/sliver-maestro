import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    def next_batch(self,batch_size,shuffle = False):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples) 
            #np.random.shuffle(idx)  
            self._data = self.data[idx]  

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples) 
            #np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch
            data_new_part =  self._data[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]

##Place all the npy quickdraw files in mypath

def split_data_multiple_files(mypath):
    txt_name_list = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        if filenames != '.DS_Store':
            txt_name_list.extend(filenames)
            break

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    xtotal = []
    ytotal = []
    ###Setting value to be 80000 for the final dataset
    slice_train = int(80000/len(txt_name_list))  
    i = 0
    seed = np.random.randint(1, 10e6)

    ##Creates test/train split with quickdraw data
    for txt_name in txt_name_list:
        txt_path = mypath + txt_name
        xx = np.load(txt_path)
        try:
            xx = xx.astype('float32') / 255.    ##scale images to binary
        except AttributeError:
            pass
        try:
            y = [i] * len(xx)
            np.random.seed(seed)
            np.random.shuffle(xx)
            np.random.seed(seed)
            np.random.shuffle(y)
            xx = xx[:slice_train]
            y = y[:slice_train]
            if i != 0:
                xtotal = np.concatenate((xx,xtotal), axis=0)
                ytotal = np.concatenate((y,ytotal), axis=0)
            else:
                xtotal = xx
                ytotal = y
            i += 1
            x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.2, random_state=42)
        except TypeError:
            pass
    return x_train, x_test, y_train, y_test

def split_data(mypath):
    ##Creates test/train split with quickdraw data
    xx = np.load(mypath)
    y = [0] * len(xx)   
    try:
        xx = xx.astype('float32') / 255. 
    except AttributeError:
        pass
   
    x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
    