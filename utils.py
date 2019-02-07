import pdb 
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import os


tf.set_random_seed(1)
np.random.seed(1)

cufs_devkit_train_name = '../cufs/devkit/train.txt'
cufs_devkit_test_name = '../cufs/devkit/test.txt'
cufs_img_dir = '../cufs/imgs/'
celeba_img_dir = '../celeba/imgs/'


im_height =64
im_width = 64

celeba_im_height=64
celeba_im_width = 64
celeba_im_channels = 1


def fileparser():
    true_img_names =[]
    with open(cufs_devkit_train_name) as file:
        for line in file:
            l = line.split(" ")
            true_img_names.append(l[0])

    with open(cufs_devkit_test_name) as file:
        for line in file:
            l = line.split(" ")
            true_img_names.append(l[0])

    return true_img_names


def image_reader(true_img_names):

    true_imgs = np.zeros([len(true_img_names), im_height, im_width, 1])
    i = 0
    for file in true_img_names:
        true_imgs[i,:,:,0]  = np.asarray(Image.open(cufs_img_dir + file), dtype=np.float32)
        # Transform the image to [-1,1] range 
        true_imgs[i,:,:,0] = 2 * true_imgs[i,:,:,0]/255.0 - 1
        i+=1

    np.random.shuffle(true_imgs)

    return true_imgs

def get_celeba_dataset(batch_size=100, option=0):
    files = os.listdir(celeba_img_dir)
    files = sorted(files)
    print("Total Number of images: ", len(files))
    true_imgs =np.zeros(shape=[len(files), celeba_im_height, celeba_im_width, celeba_im_channels], dtype=np.float32)

    i=0
    #print(files)
    for file in files:
        im=Image.open(celeba_img_dir +'/'+ file)
        # true_imgs[i,:,:,0]=np.array(im, dtype=np.float32)
        img=np.array(im, dtype=np.float32)
        true_imgs[i,:,:,0] = 2 * img/255.0 - 1
        i+=1

    
    np.random.shuffle(true_imgs)

    return true_imgs



def get_cufs_dataset(batch_size = 100, option = 0):

    true_img_names = fileparser()
    true_imgs = image_reader(true_img_names)
    if(option == 0):
        return true_imgs
    true_imgs_tensor = tf.constant(true_imgs, dtype = tf.float32)

    # Step 2: Create datasets and iterator

    data = tf.data.Dataset.from_tensor_slices((true_imgs_tensor))
    
    #data = data.shuffle(dataset_size)
    true_imgs_batch = data.batch(batch_size)
    
    return true_imgs_batch
    



def conv_bn_leaky_relu(scope_name, input, filter, k_size, stride=(1,1), padd='SAME'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        conv = tf.layers.conv2d(inputs=input,
                                filters=filter,
                                kernel_size=k_size,
                                strides=stride,
                                padding=padd)
        
        batch_norm=tf.layers.batch_normalization(inputs=conv, training=True)
        
        a =tf.nn.leaky_relu(batch_norm, name=scope.name)

    return a

def transpose_conv_bn_relu(scope_name, input, filter, k_size, stride=(1,1), padd='VALID'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        tr_conv = tf.layers.conv2d_transpose(input, filter, k_size, stride, padd, activation=None)
        
        batch_norm=tf.layers.batch_normalization(inputs=tr_conv, training=True)
        
        a =tf.nn.relu(batch_norm, name=scope.name)

    return a


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


#tf.reset_default_graph()
#imgs = get_cufs_dataset(100)
#print(imgs)