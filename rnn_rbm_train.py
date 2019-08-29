import time
import sys
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rnn_rbm
import ngsim_manipulation 

"""
    This file contains the code for training the RNN-RBM
"""

learningRate = 0.1
saved_initial_weights_path = "parameter_checkpoints/initialized.ckpt" #The path to the initialized weights checkpoint file
#saved_initial_weights_path = "parameters/epoch_5.ckpt" #The path to the initialized weights checkpoint file

def main(num_epochs):
    print("reading data...")
    data=ngsim_manipulation.Data()
    #First, we build the model and get pointers to the model parameters
    x, cost, reconstruction, W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, lr, u0 = rnn_rbm.rnnrbm()

    #The trainable parameters, as well as the initial state of the RNN
    params = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]
    opt_func = tf.train.AdamOptimizer(learning_rate=lr) 
    grad_and_params = opt_func.compute_gradients(cost, params)
    grad_and_params = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grad_and_params]
    updt = opt_func.apply_gradients(grad_and_params)
    
    #The learning rate of the  optimizer is a parameter that we set on a schedule during training
    #opt_func = tf.train.GradientDescentOptimizer(learning_rate=lr)
    #grad_and_params = opt_func.compute_gradients(cost, params)
    #grad_and_params = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grad_and_params] #We use gradient clipping to prevent gradients from blowing up during training
    #updt = opt_func.apply_gradients(grad_and_params)

    saver = tf.train.Saver(params, max_to_keep=1) #We use this saver object to restore the weights of the model and save the weights every few epochs
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init) 
        saver.restore(sess, saved_initial_weights_path) #Here we load the initial weights of the model that we created with weight_initializations.py

        ##这边，可以考虑batch，也可以一次就只处理一个traj 随机梯度下降
        print ("training in progress...")
        for epoch in range(num_epochs):
            costs = []
            start = time.time()
            #for j in tqdm(range(100)):
            for j in tqdm(range(data.num_trajectories)):
                _, C = sess.run([updt, cost], feed_dict={x: data.next_traj(), lr: learningRate }) 
                costs.append(C) 

            #Print the progress at epoch
            print ("epoch: {} cost: {} time: {}".format(epoch, np.mean(costs), time.time()-start))
            print ("\n")
            #Here we save the weights of the model every few epochs
            saver.save(sess, "parameter_checkpoints/epoch_{}.ckpt".format(epoch))

if __name__ == "__main__":
    main(int(sys.argv[1]))


