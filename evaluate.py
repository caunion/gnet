#coding: utf-8

import os
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P


def evaluate(net_file, weights, niter, feat_layer_name):
    net = caffe.Net(net_file, weights, caffe.TEST)
    data = []
    label = []
    pred = []
    for i in range(niter):
        net.forward()
        data.extend(net.blobs['data'].data.copy())
        label.extend(net.blobs['label'].data.copy())
        pred.extend(net.blobs[feat_layer_name])
    return data, label, pred



def analysis(data, label, pred, dataset_info):
    pass

def main():
    net_file = "googlenet_proto/train_val.prototxt"
    weights = "googlenet_proto/models/_iter_XXX.caffemodel"
    net_test_batch = 5
    net_feat_layer = "loss2/feat"
    test_label_file = "data/test.txt"
    data_scale_file  = "data/dataset_info.dat"
    
    test_labels = np.loadtxt(test_label_file, str, " ")
    dataset_info = np.load(data_scale_file)
    data, label, pred = evaluate(
        net_file, 
        weights, 
        len(test_labels) / net_test_batch,
        net_feat_layer)


if __name__ == "__main__":
    main()
    

