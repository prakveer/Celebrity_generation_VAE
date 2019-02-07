#VAE on CUFS dataset
#Collaborated with Abishek Tiwari


from __future__ import print_function
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

ndf = 128# determines no of channels for encoder (see encoder)
nc = 1 # no of channels in the decoder. (in our case, black and white image - 1 channel)
nz = 128 #latent dimension
lr = 0.001 # for adam optimiser
beta1=0.5 # for adam optimiser
ngf = 32 # parameter for no of channels in decoder


def plotter(entry, namer):
        #plotting
    plt.plot(range(len(entry)), entry)
    plt.ylabel(namer)
    plt.xlabel('iterations')
    plt.show()
    return
  
kl_loss=[]
recon_loss=[]
total_loss=[]



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
        log_sigma_sq = tf.layers.dense(x, nz, activation=None, name='fc_sigma')
        log_sigma_sq = tf.layers.batch_normalization(log_sigma_sq, training = training)
        log_sigma_sq = tf.nn.softplus(log_sigma_sq)

        # return mean and variance
        return mu, log_sigma_sq


def decoder(z, training, scope):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        
        
        z = tf.layers.dense(z, 256*4*4, activation=None, name='decoder_fc')
        z = tf.reshape(z, [-1, 4, 4, 256]) 
        z = tf.layers.batch_normalization(z, training = training)

        
        
        tc = tf.layers.conv2d_transpose(z,filters=ngf * 4, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())
        tc = tf.layers.batch_normalization(tc, training = training)
        tc = tf.nn.relu(tc)
        #print('tc1', tc)

        tc = tf.layers.conv2d_transpose(tc,filters=ngf * 2, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())
                                        
        tc = tf.layers.batch_normalization(tc, training = training)
        tc = tf.nn.relu(tc)
        
        #print('tc2', tc)

        ##########       
        tc = tf.layers.conv2d_transpose(tc,filters=ngf, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())
                                      
        tc = tf.layers.batch_normalization(tc, training = training)
        tc = tf.nn.relu(tc)
        
        #print('tc3', tc)

        
        #######################
        tc = tf.layers.conv2d_transpose(tc,filters=nc, kernel_size=[5,5], 
                                        strides=(2, 2), padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0, 0.02), 
                                        bias_initializer =tf.zeros_initializer())
       
        #print('tc4', tc)
                
        
        return tf.nn.tanh(tc)


class VAE(object):

    def __init__(self):
        self.lr = lr
        self.beta1 = beta1
        self.batch_size = 32
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        
        self.skip_step = 10
        self.training=True

    def get_data(self):
        with tf.name_scope('data'):
            print("loading data")
            self.true_imgs_dataset = utils.get_cufs_dataset( option =0)
            print(self.true_imgs_dataset.shape)
            self.img =  tf.placeholder(dtype = tf.float32, shape=[None, 64, 64, 1])

    def inference(self):
        #reconstruction 
        self.q_mu, self.q_log_sigma_sq = encoder(self.img, self.training, 'encoder')
        epsilon = tf.random_normal([self.batch_size, nz],mean=0.0, stddev=1.0)
        std = tf.exp(.5*self.q_log_sigma_sq)
        z = self.q_mu + std *epsilon
        self.x_recon = decoder(z, self.training, 'decoder')
        
        #addition of 2 images in latent space
        ind = tf.random.uniform([2],minval=0,maxval=self.batch_size-1,dtype=tf.int32, seed = 117)
        z_mix1 = .5 * z[ind[0],:] + .5*z[ind[1],:]
        z_mix1 = tf.reshape(z_mix1, [1,-1])
        self.x_mix1 = decoder(z_mix1, self.training, 'decoder')

        z_mix2 = .3 * z[ind[0],:] + .7*z[ind[1],:]
        z_mix2 = tf.reshape(z_mix2, [1,-1])
        self.x_mix2 = decoder(z_mix2, self.training, 'decoder')

        z_mix3 = .7 * z[ind[0],:] + .3*z[ind[1],:]
        z_mix3 = tf.reshape(z_mix3, [1,-1])
        self.x_mix3 = decoder(z_mix3, self.training, 'decoder')

        self.mix1 = self.img[ind[0],:,:,:]
        self.mix2 = self.img[ind[1],:,:,:]
       
        
    def loss(self):
            with tf.variable_scope('losses', reuse=tf.AUTO_REUSE) as scope:
                # self.loss_recon = tf.reduce_sum( tf.losses.mean_squared_error(labels=self.img, predictions=self.x_recon) )
                self.loss_recon = tf.reduce_mean(tf.square(self.x_recon - self.img))
                var = tf.exp(self.q_log_sigma_sq)
                self.KL = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.q_log_sigma_sq - tf.square(self.q_mu) - var,1))
                self.loss = self.loss_recon +  5 * 1e-3 * self.KL


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
            tf.summary.scalar('KL', self.KL)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('loss_recon', self.loss_recon)
            self.summary_op = tf.summary.merge_all()                          

    def build(self):
        self.get_data()
        self.inference()
        self.gen_from_noise()
        self.loss()
        self.optimize()
        self.summary()
       

    def train_one_epoch(self, sess, saver, writer, epoch, step):
        start_time = time.time()
        self.training = True

        for i in range (int( (1*len(self.true_imgs_dataset) )/ self.batch_size)):
            if((i+1) * self.batch_size < len(self.true_imgs_dataset)):
                st = i * self.batch_size
                en = (i+1) * self.batch_size
            else:
                st = i * self.batch_size
                en = len(self.true_imgs_dataset)   

            _, loss, loss_recon, kl_out = sess.run([self.opt, self.loss, self.loss_recon, self.KL], {self.img: self.true_imgs_dataset[st: en]})#, bar2])
               
            kl_loss.append(kl_out)
            recon_loss.append(loss_recon)
            total_loss.append(loss)
            
            
            
            # if (step + 1) % self.skip_step == 0:
            #     print('Loss at step {0}: gen: {1}, KL: {2}, lrecon: {3}'.format(step, loss, kl_out, loss_recon))
            #     fake_out, img = sess.run([self.x_recon, self.img], {self.img: self.true_imgs_dataset[st: en]})  
            #     x_recon_from_noise = sess.run(self.x_recon_from_noise, {self.img: self.true_imgs_dataset[st: en]}) 

            #     for j in range(2):
            #         i=random.randint(1, self.batch_size-1)
            #         self.showImg(img, fake_out, x_recon_from_noise, i)

                  
            step += 1



        return step

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('vae')
        utils.safe_mkdir( '../../checkpoints')
        utils.safe_mkdir('../../checkpoints/convnet_layers')
        writer = tf.summary.FileWriter('../.././graphs/convnet_layers', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, writer, epoch, step)
                
            # To display reconstructed images and random generated images
            st =random.randint(1, self.true_imgs_dataset.shape[0])
            fake_out, img = sess.run([self.x_recon, self.img], {self.img: self.true_imgs_dataset[st:st+self.batch_size]})  
            x_recon_from_noise = sess.run(self.x_recon_from_noise, {self.img: self.true_imgs_dataset[st: st+self.batch_size]}) 

            #to display addition of 2 images in latent space
            for j in range(4):
                i=random.randint(1, self.batch_size-1)
                self.showImg(img, fake_out, x_recon_from_noise, i)
                x_mix1, x_mix2, x_mix3, mix1, mix2 = sess.run([self.x_mix1, self.x_mix2, self.x_mix3 , self.mix1, self.mix2], {self.img: self.true_imgs_dataset[st:st+self.batch_size]})
                # pdb.set_trace()  
                self.showImg(mix1.reshape(1,64,64,1), mix2.reshape(1,64,64,1), x_mix1, 0)
                self.showImg(mix1.reshape(1,64,64,1), mix2.reshape(1,64,64,1), x_mix2, 0)
                self.showImg(mix1.reshape(1,64,64,1), mix2.reshape(1,64,64,1), x_mix3, 0)

                # self.showImg(img, i)
                # self.showImg(x_recon_from_noise, i)

            #pdb.set_trace()
            #plt.show()
        writer.close()

    def showImg(self, im, fake, infer, i):
        
        im = np.reshape(im[i,:,:,0], [64,64])
        im = ((im+1)*255.0)/2

        f = np.reshape(fake[i,:,:,0], [64,64])
        f = ((f+1)*255.0)/2

        infer = np.reshape(infer[i,:,:,0], [64,64])
        infer = ((infer+1)*255.0)/2


        plt.figure(1)
        plt.subplot(131)
        plt.imshow(im, cmap='gray')

        plt.subplot(132)
        plt.imshow(f, cmap='gray')

        plt.subplot(133)
        plt.imshow(infer, cmap='gray')
        plt.show()

    def imshow(self, im):
        im = np.reshape(im[0,:,:,0], [64,64])
        im = ((im+1)*255.0)/2
        plt.imshow(im, cmap='gray')
        plt.show()

    def saveImg(self, im, i, name):
        
        im = np.reshape(im[i,:,:,0], [64,64])
        im = ((im+1)*255.0)/2
        plt.imsave(name, im.astype(np.uint8), cmap='gray')

    def gen_from_noise(self):
        # generate from noise
        noise = tf.random_normal([self.batch_size, nz],mean=0.0, stddev=1.0)
        self.x_recon_from_noise = decoder(noise, self.training, 'decoder')


if __name__ == '__main__':
    tf.reset_default_graph()
    model = VAE()
    model.build()
    print('model constructed, starting to train!')
    model.train(n_epochs=5)
    plotter(kl_loss, "KL LOSS")
    plotter(recon_loss, "L2 reconstructing LOSS")
    plotter(total_loss, "TOTAL LOSS")