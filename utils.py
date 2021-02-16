import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
from skimage import feature, color


def list_all_files(dir_name):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dir_name)
    allFiles = []
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + list_all_files(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


def edge_filename(filename):
    """
    Creating new filename for extracted edges image
    """
    fspl = filename.split('/')
    imagename = fspl[-1]
    newfile = '.'.join(imagename.split('.')[:-1] + ['edges', 'jpg'])
    return '/'.join(fspl[:-1] + [newfile])


def image_to_edges(filename, sigma=1.12):
    """
    Retrieve edges from the image
    """
    edges = np.asarray(Image.open(filename))
    edges = color.rgb2gray(edges)
    edges = np.logical_not(feature.canny(edges, sigma=sigma)).astype(np.float32) * 255.
    return edges


def show(source_image, target_image):
    
    plt.figure(figsize=(6, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Source Image')
    plt.imshow((source_image.numpy()[:, :, 0] + 1) / 2, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Target Image')
    plt.imshow((target_image.numpy() + 1) / 2)
    plt.axis('off')
         

def show_tf_batch(tf_batch, imtype='rgb', n_img = 25, to_print=None):
    images = tf_batch.numpy()
    images = (images + 1) / 2
    sqrtn = int(np.ceil(np.sqrt(n_img)))
    # select subset
    images = images[:n_img]
    # set grid
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')        
        if imtype == 'rgb':
            plt.imshow(img)
        else:
            plt.imshow(img[:, :, 0], cmap='gray')
    if to_print is not None:
        print(to_print)
    return
