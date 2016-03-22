#coding: UTF-8
import numpy as np
import shutil
import h5py
import os
import glob
import lmdb
import caffe
import threading
import multiprocessing
from PIL import Image
from os import path, system

def lmdb2label(dbname="all_db"):
    env = lmdb.open(dbname, readonly=True)
    data = []
    files = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            flat_x = np.fromstring(datum.data, dtype=np.float32)
            x = flat_x.reshape((1,1,6))
            y = datum.label
            data += flat_x,
            files += key,
    data = np.asarray(data)
    files = np.asarray(files)
    return files, data

def label2lmdb(files, data, filename="all_info"):
    map_size = data.nbytes * 10 + files.nbytes * 20
    env = lmdb.open(filename, map_size = map_size)
    N = len(data)
    with env.begin(write=True) as txn:
        for i in range(N):
            item = data[i]
            fname = files[i]
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = 1
            datum.width = 6
            datum.data = item.tobytes()
            datum.label = 88888
            #str_id = "Screenshot{:08}".format(int(fname[len('Screenshot'):]))
            str_id = "{:08}".format(i+1)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

    return True

def label2hdf5(data, filename = 'all_info.h5'):
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
            image_path = path.join(prefix, path.basename(f))
            if not path.exists(image_path):
                print "image {0} doesn't exists..".format(image_path)
            else:
                fid.write(image_path + " 9999999\n")
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

def readinfotxt(filename = "info.txt", shuffle = False):
    files =[]
    data = []
    with open(filename, 'r') as fid:
        lines = fid.readlines()
        if shuffle:
            np.random.shuffle(lines)
        for line in lines:
            if line is None or line == "":
                continue
            fname, x, y, z, roll, pitch, yaw = line.split(' ')
            x = float(x)
            y = float(y)
            z = float(z)
            roll = float(roll) if float(roll) < 180.0 else (float(roll) - 360.0)
            roll = roll * (2.0 * np.pi) / 360.0
            pitch = float(pitch) * (2.0 * np.pi) / 360.0
            yaw = float(yaw) * (2.0 * np.pi) / 360.0
            data.append([x, y, z, roll, pitch, yaw])
            fname = "Screenshot{0}".format(fname[len('Screenshot00'):])
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
    if not path.exists(target_folder):
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


def build_data_set(image_folder, shuffule, suffix="", keep_dim=[]):
    #shutil.rmtree("data/")
    save_folder = "data" + "_" + str(suffix)
    if not path.exists(save_folder):
        os.makedirs(save_folder)
    # os.system('mv data data_' + suffix)
    # os.makedirs("data")
    files, data = readinfotxt(path.join(image_folder, "info.txt"), shuffle=shuffule)
    maxv, minv, norm_data = norm_by_dim(data, keep_dim=keep_dim)
    train_files, train_data, test_files, test_data = monkey_split_train_test(files, norm_data, c= 20)
    label2lmdb(train_files, train_data, save_folder+'/train')
    files2txt(
        train_files,
        prefix=image_folder,
        filename = save_folder+ "/train.txt")
    label2lmdb(test_files, test_data, save_folder+'/test')
    files2txt(
        test_files,
        prefix=image_folder,
        filename=save_folder+"/test.txt")

    dataset_info = save_folder+"/dataset_info.dat"
    with open(dataset_info, 'wb') as fid:
        np.save(fid, (maxv, minv, train_files, test_files))

    train_files_read, train_data_read = lmdb2label(save_folder+'/train')
    test_files_read, test_data_read = lmdb2label(save_folder+'/test')

    ntrain, ntest = len(train_data), len(test_data)
    assert abs(sum(sum(train_data == train_data_read)) / 6 - ntrain * 1.0) < 1e-5
    assert abs(sum(sum(test_data == test_data_read)) / 6 - ntest*1.0) < 1e-5

def train_net(solver_path, param="", prefix=""):
    folder=path.dirname(solver_path)
    with open(solver_path, 'r') as fid:
        lines = fid.readlines()
        lines[-1] = lines[-1][0: lines[-1].rfind('/')] + "/" + prefix + "\""
    with open(solver_path, 'w') as fid:
        content = "".join(lines)
        fid.write(content)
    os.system("/home/daoyuan/caffe/build/tools/caffe train --solver={0} {1} -gpu 0"
         .format(
             solver_path, 
             param
         ))
    pass
def main():
    caffe_net = "caffenet_proto/solver.prototxt"
    caffe_net_model="--weights=caffenet_proto/bvlc_reference_caffenet.caffemodel"
    caffe_net_solverstate="--snapshot=caffenet_proto/models/_iter_21028.solverstate"

    ## shuffled dataset
    train_net(caffe_net, caffe_net_solverstate, prefix="")
    train_net(caffe_net, prefix="2")
    train_net(caffe_net, param=caffe_net_model, prefix="3")

    #build unshuffled dataset
    build_data_set(False)
    train_net(caffe_net, prefix="4")
    train_net(caffe_net, param=caffe_net_model, prefix="5")

if __name__ == "__main__":
    #main()
    source_folder = "/media/daoyuan/My Passport/stadium_rand/"
    target_folder ="/media/hdd3/stadium_rand_resize/"
    #resize( source_folder, 
    #        target=(244, 244), 
    #        target_folder=target_folder,
    #        nthreads=6
    #        )
    build_data_set(target_folder, True, suffix="rand_monk")
     
    
    # image_folder = "/media/xiaocan/statium_image_data/"
    # crop(image_folder,
    #        target=(244,244),
    #        target_folder= ("/media/daoyuan/hdd1/daoyuan/stadim_image_crop"),
