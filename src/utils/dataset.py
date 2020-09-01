import numpy as np
from sklearn.model_selection import train_test_split
from os import walk
import os


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


# Place all the npy quickdraw files in mypath
# MC: Do we need this function?
def split_data_multiple_files(mypath, category=False, shuffle=False):
    txt_name_list = []
    if category:
        mypath = 'data/input/' + category + '/'
        for (dirpath, dirnames, filenames) in walk(mypath):
            for filename in filenames:
                if filename.endswith('.npy'):
                    txt_name_list.append(filename)
                    break
    else:
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
    # Setting value to be 80000 for the final dataset
    slice_train = int(80000 / len(txt_name_list))
    i = 0
    seed = np.random.randint(1, 10e6)

    # Creates test/train split with quickdraw data
    for txt_name in txt_name_list:
        txt_path = mypath + txt_name
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


# Place all the npy quickdraw files in mypath
def split_data(mypath, category, shuffle=False):
    # Creates test/train split with quickdraw data
    mypath = os.path.join(mypath, category + '/full_numpy_bitmap_%s.npy' % category) 
    xx = np.load(mypath)
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


