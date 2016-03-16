import numpy as np
import h5py
import os
import glob
from PIL import Image
from os import path, system

def label2lmdb(data, filename = 'all_info.h5'):
    rows, channels, height, width = len(data), 1, 1, 6
    total_size =  rows * channels * height * width
    reshape_mat = np.arange( total_size).astype('float32')
    reshape_mat = reshape_mat.reshape(rows, channels, height, width)
    reshape_mat[:,0,0,:] = data
    with h5py.File(filename, 'w') as f:
        f['data'] = reshape_mat
    return True

def norm_by_dim(data, keep_dim = []):
    rows = len(data)
    maxv = np.max(data, axis = 0)
    minv = np.min(data, axis = 0)
    maxv[keep_dim] = 1
    minv[keep_dim] = 0
    rangev= maxv -minv
    rangev[ rangev == 0 ] = 1.0
    ret = (data - np.tile(minv, (rows, 1))) / np.tile(rangev, (rows, 1))
    return maxv, minv, ret

def readinfotxt(filename = "info.txt.bak"):
    files =[]
    data = []
    with open(filename, 'r') as fid:
        while True:
            line = fid.readline()
            if line is None or line == "": break
            fname, x, y, z, roll, pitch, yaw = line.split(' ')
            x = float(x)
            y = float(y)
            z = float(z)
            roll = float(roll) if float(roll) < 180.0 else (float(roll) - 360.0)
            roll = roll * (2.0 * np.pi) / 360.0
            pitch = float(pitch) * (2.0 * np.pi) / 360.0
            yaw = float(yaw) * (2.0 * np.pi) / 360.0
            data.append([x, y, z, roll, pitch, yaw])
            files.append(fname)
    data = np.asarray(data).astype('float32')
    files = np.asarray(files)
    return files, data
def split_train_test(files, data, ratio = 0.1):
    n = len(files)
    I = np.random.choice(n, n*ratio, replace = False )
    I_train = list(set(range(n)) - set(I))
    test_files = files[I]
    test_data = data[I]
    train_files = files[I_train]
    train_data = data[I_train]
    return train_files, train_data, test_files, test_data
def monkey_split_train_test(files, data, c = 1):
    n = len(files)
    group_size = 133
    group_offset = np.random.choice(group_size, c,replace=False)
    group_offset.sort()
    I_test = np.asarray( [ (group_offset + (i* group_size)) for i in range(n / group_size )] )
    I_test = I_test.flatten()
    I_train = list(set(range(n)) - set(I_test))
    test_files, test_data  = files[I_test],  data[I_test]
    train_files, train_data = files[I_train], data[I_train]
    return train_files, train_data, test_files, test_data

def resize(folderpath, target=(326,244), target_folder ='resized'):
    filelist = glob.glob(path.join(folderpath,'*.jpg'))
    #target_folder = path.join(folderpath,'..',target_folder)
    system('mkdir ' + target_folder)
    for file in filelist:
        im = Image.open(file)
        out = im.resize(target)
        out.save(path.join(target_folder, path.basename(file)))

def main():
    files, data = readinfotxt()
    maxv, minv, norm_data = norm_by_dim(data, keep_dim=[0,1,2])
    train_files, train_data, test_files, test_data = monkey_split_train_test(files, norm_data, c= 10)
    label2lmdb(train_data, 'train.h5')
    label2lmdb(test_data, 'test.h5')

if __name__ == "__main__":
    main()