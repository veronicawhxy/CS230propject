import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image

#Convert tif files to jpg files

def convertImageFormat(root): 
    extra_counter = 0
    for infile in os.listdir(root):
        # print(os.path.join(dir_path, infile))
        if infile.endswith('.tif') or infile.endswith('.TIF'):
            if os.path.isfile(os.path.splitext(os.path.join(root, infile))[0] + ".jpg"):
                    print("A jpg file already exists for %s" % infile)
                    outfile = infile[:-4] + "_" + extra_counter + ".jpg"
                    extra_counter += 1
            else: 
                outfile = infile[:-3] + "jpg"
                im = Image.open(os.path.join(root, infile))
                out = im.convert("RGB")
                # outputfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                out.save(os.path.join(root, outfile), "JPEG", quality=90)
                os.remove(os.path.join(root, infile))

convertImageFormat(root = "dataset/unzip/CASIA2/Tp")

import random
import shutil

def split_dataset(root, c, dest):
    # root = "dataset/unzip/Au"
    files = os.listdir(root)
    k = c * (len(files))
    indices = random.sample(range(0, len(files)), int(k))
    for i in indices:
        selected = files[i]
        shutil.move(os.path.join(root, selected), os.path.join(dest, selected))
    
split_dataset(root = "./dataset/unzip/CASIA2/Au", c = 0.05, dest = "./dataset/dev/Au")
split_dataset(root = "./dataset/unzip/CASIA2/Tp", c = 0.05, dest = "./dataset/dev/Tp")
split_dataset(root = "./dataset/unzip/CASIA2/Au", c = 0.05, dest = "./dataset/test/Au")
split_dataset(root = "./dataset/unzip/CASIA2/Tp", c = 0.05, dest = "./dataset/test/Tp")

# Check divided dataset¶
files = os.listdir("./dataset/unzip/CASIA2/Au")
len(files)