import os
import os.path as path
import numpy as np
import glob
from PIL import Image

def PIL2mat(image):
    data = np.asarray(image.getdata(), dtype=np.uint8).\
        reshape((image.size[0], image.size[1], 3))
    return data
def single_img_mean(image_path):
    if not path.exists(image_path):
        raise Exception('image not exists')
    im = Image.open(image_path,'r')
    data = PIL2mat(im)
    ret = np.mean(data, axis=(0,1)) # R, G, B
    return ret

def image_mean(img_list):
    ret = np.array([0.0,0.0,0.0])
    for i, f in enumerate(img_list):
        ret += single_img_mean(f)
        if i % 10000 == 0:
            print "processed %d images" % (i+1)
    ret /= len(img_list)
    return ret
def main():
    folder = "/"
    img_list = glob.glob(folder)
    ret = image_mean(img_list)
    print "mean value: {0}".format(ret)

if __name__ == "__main__":
    main()