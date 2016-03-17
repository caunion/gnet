#charset: UTF-8
import numpy as np
import h5py
import os
import glob
import threading
import multiprocessing
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

def files2txt(files, prefix = "", filename = "all_files.txt"):
    with open(filename, 'w') as fid:
        for f in files:
            fid.write(
                path.join(
                    prefix,
                    path.basename(f)
                ) + "\n"
            )
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

def readinfotxt(filename = "info.txt"):
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

def crop(folderpath, target=(244,244), target_folder = "cropped", crop_center = True, nthreads=4):
    if not path.exists(target_folder):
        os.makedirs(target_folder)
    threads = []
    filelist = glob.glob(path.join(folderpath, "*.jpg"))
    filelist.sort()
    nfiles = len(filelist)
    ntasksize = nfiles / nthreads
    print "start {0} threads to crop {1} files".format(nthreads, nfiles)
    def crop_list(imagelist):
        for f in imagelist:
            img = Image.open(f)
            w, h = img.size[0], img.size[1]
            w_o, h_o = target
            out = img.crop(
                ( (w - w_o) / 2,
                  (h - h_o) / 2,
                  (w + w_o) / 2 ,
                  (h + h_o) / 2,
                )
            )
            out.save( path.join(
                target_folder,
                path.basename(f)
            ))
    for i in range(nthreads):
        left, right = i * ntasksize, min( (i+1) * ntasksize + nthreads, nfiles)
        thread = multiprocessing.Process(
            target=crop_list,
            args=(filelist[left:right], )
        )
        threads += thread,
        print "start thread {0} for image {1} to {2}".format(i, left, right)
        thread.start()
    for t in threads:
        t.join()
    print "finished crop, total {0} images".format(nfiles)

def resize(folderpath, target=(326,244), target_folder ='resized', nthreads = 4):
    if not path.exists(folderpath):
        os.makedirs(target_folder)
    threads = []
    filelist = glob.glob(path.join(folderpath,'*.jpg'))
    filelist.sort()
    nfiles = len(filelist)
    ntasksize = nfiles/nthreads
    def resize_list(filelist):
        for file in filelist:
            im = Image.open(file)
            out = im.resize(target)
            out.save(path.join(target_folder, path.basename(file)))
    for i in range(nthreads):
        left, right = i*ntasksize, min((i+1)*ntasksize+ nthreads, nfiles)
        thread = multiprocessing.Process(
                    target=resize_list,
                    args=(filelist[left: right],)
        )
        threads.append(thread)
        print "start thread {0} for image {1} to {2}".format(
            i, left, right
        )
        thread.start()

    for t in threads:
        t.join()
    print "finished resize"


def main():
    files, data = readinfotxt()
    maxv, minv, norm_data = norm_by_dim(data, keep_dim=[0,1,2])
    train_files, train_data, test_files, test_data = monkey_split_train_test(files, norm_data, c= 10)
    label2lmdb(train_data, 'train.h5')
    files2txt(
        train_files,
        prefix="/media/daoyuan/hdd1/daoyuan/stadim_image_crop/",
        filename = "train.txt")
    label2lmdb(test_data, 'test.h5')
    files2txt(
        test_files,
        prefix="/media/daoyuan/hdd1/daoyuan/stadim_image_crop/",
        filename="test.txt")

    dataset_info = "dataset_info.dat"
    with open(dataset_info, 'wb') as fid:
        np.save(fid, (maxv, minv, train_files, test_files))


if __name__ == "__main__":
    main()
    # image_folder = "/media/xiaocan/statium_image_data/"
    # crop(image_folder,
    #        target=(244,244),
    #        target_folder= ("/media/daoyuan/hdd1/daoyuan/stadim_image_crop"),