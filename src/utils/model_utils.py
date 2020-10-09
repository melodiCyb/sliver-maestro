import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import subprocess
from google.cloud import storage
from configparser import ConfigParser
import sys

base_path = os.getcwd().split('sliver-maestro')[0]
base_path = os.path.join(base_path, "sliver-maestro")
sys.path.insert(1, base_path)

config = ConfigParser()
cfg_file = os.path.join(base_path, "src/config.cfg")
config.read(cfg_file)

credentials = config['GOOGLE_APPLICATION_CREDENTIALS']['credentials']
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=credentials

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
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]


# Place all the npy quickdraw files in filepath
def split_data(filepath, category, shuffle=False):
    # Creates test/train split with quickdraw data
    filepath = os.path.join(filepath, category, '%s.npy' % category)
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

def create_folders(root_path, category):

    data_path = os.path.join(root_path, "src", "data")
    save_path = os.path.join(root_path, "src", "save")
    input_path = os.path.join(data_path, "input")
    output_path = os.path.join(data_path, "output")
    images_path = os.path.join(output_path, "images")
    positions_path = os.path.join(output_path, "positions")
    raw_path = os.path.join(data_path, "raw")
    category_save_path = os.path.join(save_path, category)
    category_input_path = os.path.join(data_path, "input", category)
    category_raw_path = os.path.join(data_path, "raw", category)
    category_images_path = os.path.join(images_path, category)
    base_input = os.path.join(category_input_path, category)
    base_raw = os.path.join(category_raw_path, category)
    
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
    
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        if os.path.exists(data_path):
            print("{} folder is created...".format(data_path))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        if os.path.exists(save_path):
            print("{} folder is created...".format(save_path))
            
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        if os.path.exists(input_path):
            print("{} folder is created...".format(input_path))        
            
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        if os.path.exists(output_path):
            print("{} folder is created...".format(output_path))

    if not os.path.exists(images_path):
        os.mkdir(images_path)
        if os.path.exists(images_path):
            print("{} folder is created...".format(images_path))
            
    if not os.path.exists(positions_path):
        os.mkdir(positions_path)
        if os.path.exists(positions_path):
            print("{} folder is created...".format(positions_path))
            
    if not os.path.exists(raw_path):
        os.mkdir(raw_path)
        if os.path.exists(raw_path):
            print("{} folder is created...".format(raw_path))

    if not os.path.exists(category_save_path):
        os.mkdir(category_save_path)
        if os.path.exists(category_save_path):
            print("{} folder is created...".format(category_save_path))   
            
    if not os.path.exists(category_input_path):
        os.mkdir(category_input_path)
        if os.path.exists(category_input_path):
            print("{} folder is created...".format(category_input_path))     
    
    if not os.path.exists(category_raw_path):
        os.mkdir(category_raw_path)
        if os.path.exists(category_raw_path):
            print("{} folder is created...".format(category_raw_path))

    if not os.path.exists(category_images_path):
        os.mkdir(category_images_path)
        if os.path.exists(category_images_path):
            print("{} folder is created...".format(category_images_path))
 
    print("folders are created...")

    src_list = []
    dst_list = []

    src_file = "full/numpy_bitmap/" + category + ".npy"
    #src_file = "gs://quickdraw_dataset/full/numpy_bitmap/" + categories[category] + ".npy"
    src_list.append(src_file)

    #src_file = "gs://quickdraw_dataset/full/raw/" + categories[category] + ".ndjson"
    src_file = "full/raw/" + category + ".ndjson"
    src_list.append(src_file)

    dst_file_input = os.path.join(category_input_path, category)
    dst_file_raw = os.path.join(category_raw_path, category)

    dst_list.append(dst_file_input + ".npy")
    dst_list.append(dst_file_raw + ".ndjson")    
    
    return src_list, dst_list, paths_dict


def download_blob(bucket_name, source_blob_name, destination_file_name):
    
    """Downloads a blob from the bucket."""
    bucket_name = "quickdraw_dataset"
    source_blob_name = source_blob_name
    destination_file_name = destination_file_name

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def download_data(src_list, dst_list):

    print('Credendtials from environ: {}'.format(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))    
    
    for (src_file, dst_file) in zip(src_list, dst_list):
        bucket_name = "your-bucket-name"
        source_blob_name = src_file
        destination_file_name = dst_file        
        download_blob(bucket_name, source_blob_name, destination_file_name)

# ---------------
# model_utils.py
# ---------------

def matmul(X,Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i],Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)



