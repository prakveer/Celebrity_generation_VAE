#autoencoder on CUFS dataset

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import time 
import random
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pdb
import utils

ndf = 32
nc = 1
# nz: latent dimension
nz = 64
lr = 0.001
beta1 = 0.5

def encoder(img, training, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        x = tf.layers.conv2d(inputs=img, name='conv1', reuse=tf.AUTO_REUSE,
                             filters=ndf,
                             kernel_size=[5, 5],
                             strides=(2, 2),
                             padding='SAME',
                             kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                             bias_initializer =tf.zeros_initializer(),
                             activation= None)

        x = tf.layers.batch_normalization(x, training = training)
        x = tf.nn.relu(x, name='leaky_relu1')

        x = tf.layers.conv2d(inputs=x, name='conv2',reuse=tf.AUTO_REUSE,
                             filters=ndf * 2,
                             kernel_size=[5, 5],
                             strides=(2, 2),
                             padding='SAME',
                             kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                             bias_initializer =tf.zeros_initializer(),
                             activation= None)
        x = tf.layers.batch_normalization(x, training = training)
        x = tf.nn.relu(x, name='lr2')

        x = tf.layers.conv2d(inputs=x, name='conv3',reuse=tf.AUTO_REUSE,
                             filters=ndf * 4,
                             kernel_size=[5, 5],
                             strides=(2, 2),
                             padding='SAME',
                             kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                             bias_initializer =tf.zeros_initializer(),
                             activation= None)                            
        x = tf.layers.batch_normalization(x, training = training)
        x = tf.nn.relu(x, name='lr3')

        x = tf.layers.conv2d(inputs=x, name='conv4',reuse=tf.AUTO_REUSE,
                             filters=ndf * 8,
                             kernel_size=[5, 5],
                             strides=(2, 2),
                             padding='SAME',
                             kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                             bias_initializer =tf.zeros_initializer(),
                             activation= None)                            
        x = tf.layers.batch_normalization(x, training = training)
        x = tf.nn.relu(x, name='lr4')        
        
        dim = x.shape[1] * x.shape[2] * x.shape[3]
        x = tf.reshape(x, [-1, dim])
        mu = tf.layers.dense(x, nz, activation=None, name='fc_mu')
        mu = tf.layers.batch_normalization(mu, training = training)


        
        # The standard deviation must be positive. Parametrize with a softplus
        # log_sigma_sq = tf.layers.dense(x, nz, activation=None, name='fc_sigma')
        # log_sigma_sq = tf.layers.batch_normalization(log_sigma_sq, training = training)
        # log_sigma_sq = tf.nn.softplus(log_sigma_sq)


        return mu


def decoder(z, training, scope):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        ngf = ndf
        # pdb.set_trace()
        z =tf.reshape(z, [-1, 1, 1, z.shape[1]])
        tc = tf.layers.conv2d_transpose(z,filters=ngf * 4, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())
        tc = tf.layers.batch_normalization(tc, training = training)
        tc = tf.nn.relu(tc)
        # pdb.set_trace()
        tc = tf.layers.conv2d_transpose(tc,filters=ngf * 2, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())
                                        
        tc = tf.layers.batch_normalization(tc, training = training)
        tc = tf.nn.relu(tc)
               
        tc = tf.layers.conv2d_transpose(tc,filters=ngf, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())
                                      
        tc = tf.layers.batch_normalization(tc, training = training)
        tc = tf.nn.relu(tc)

        tc = tf.layers.conv2d_transpose(tc,filters=nc, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())

        tc = tf.layers.batch_normalization(tc, training = training)
        tc = tf.nn.relu(tc)

        tc = tf.layers.conv2d_transpose(tc,filters=nc, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())

        tc = tf.layers.batch_normalization(tc, training = training)
        tc = tf.nn.relu(tc)
        tc = tf.layers.conv2d_transpose(tc,filters=nc, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())
                                       
        # tc = tf.layers.batch_normalization(tc, training = training)
        
        print(tc)
        
        return tf.nn.tanh(tc)


class VAE(object):

    def __init__(self):
        self.lr = lr
        self.beta1 = beta1
        self.batch_size = 47
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        
        self.skip_step = 20
        self.training=True

    def get_data(self):
        with tf.name_scope('data'):
            true_imgs_dataset = utils.get_cufs_tf_dataset(self.batch_size, option =0)
            iterator = tf.data.Iterator.from_structure(true_imgs_dataset.output_types, 
                                                   true_imgs_dataset.output_shapes)
            self.img = iterator.get_next()
            self.train_init = iterator.make_initializer(true_imgs_dataset)

    def inference(self):
        self.q_mu = encoder(self.img, self.training, 'encoder')
        # pdb.set_trace()
        # epsilon = tf.random_normal([self.batch_size, nz],mean=0.0, stddev=1.0)
        # std = tf.exp(self.q_log_sigma_sq)
        z = self.q_mu 
        self.x_recon = decoder(z, self.training, 'decoder')
       
        
    def loss(self):
            with tf.variable_scope('losses', reuse=tf.AUTO_REUSE) as scope:
                loss_recon = tf.reduce_sum( tf.losses.mean_squared_error(labels=self.img, predictions=self.x_recon) )
                # var = tf.exp(self.q_log_sigma_sq)
                # KL = 0.5 * tf.reduce_sum(tf.square(self.q_mu) + var - 1 - self.q_log_sigma_sq)
                self.loss = loss_recon





    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.opt = tf.train.AdamOptimizer(self.lr, beta1 = self.beta1).minimize(self.loss,
                                                global_step=self.gstep)


    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()                         

    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.summary()
       

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        self.training = True

        try:
            while True:

                _, loss, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
               
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: gen: {1}'.format(step, loss))
                  
                step += 1

        except tf.errors.OutOfRangeError:
            pass
        sess.run(self.train_init)
        fake_out, img = sess.run([self.x_recon, self.img]) 
        i= random.randint(1, self.batch_size-1)
        # self.showImg(fake_out, i)
        # self.showImg(img, i)
        # self.saveImg(fake_out, i, "autoencoder/"+str(epoch)+"recon")
        # self.saveImg(img, i, "autoencoder/"+str(epoch)+"real")
        return step

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('autoencoder')
        utils.safe_mkdir( '../../checkpoints')
        utils.safe_mkdir('../../checkpoints/convnet_layers')
        writer = tf.summary.FileWriter('../.././graphs/convnet_layers', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                
            sess.run(self.train_init)
            fake_out, img = sess.run([self.x_recon, self.img])  
            print(fake_out.shape)
            i=random.randint(1, self.batch_size)
            self.showImg(fake_out, i)
            self.showImg(img, i)
            for j in range(10):
                i=random.randint(1, self.batch_size)
                self.saveImg(fake_out, i, "autoencoder/"+ "recon" + str(i))
                self.saveImg(img, i, "autoencoder/" + "real" + str(i))
            pdb.set_trace()
        writer.close()

    def showImg(self, im, i):
        
        im = np.reshape(im[i,:,:,0], [64,64])
        im = ((im+1)*255.0)/2
        plt.imshow(im.astype(np.uint8), cmap='gray')
        plt.show()

    def saveImg(self, im, i, name):
        
        im = np.reshape(im[i,:,:,0], [64,64])
        im = ((im+1)*255.0)/2
        plt.imsave(name, im.astype(np.uint8), cmap='gray')

if __name__ == '__main__':
    tf.reset_default_graph()
    model = VAE()
    model.build()
    model.train(n_epochs=230)