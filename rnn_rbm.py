import tensorflow as tf
import numpy as np

import RBM
import ngsim_manipulation


"""
    This file contains the TF implementation of the RNN-RBM, as well as the hyperparameters of the model
"""

n_visible          = 2   #The size of the RBM visible layer
n_hidden           = 1  #The size of the RBM hidden layer
n_hidden_recurrent = 20 #The size of each RNN hidden layer

def rnnrbm():

    #This function builds the RNN-RBM and returns the parameters of the model

    x  = tf.placeholder(tf.float32, [None, n_visible]) #The placeholder variable that holds our data
    lr  = tf.placeholder(tf.float32) #The learning rate. We set and change this value during training.
    
    size_bt = tf.shape(x)[0] 
    # parameters
    W   = tf.Variable(tf.zeros([n_visible, n_hidden]), name="W")
    Wuh = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden]), name="Wuh")
    Wuv = tf.Variable(tf.zeros([n_hidden_recurrent, n_visible]), name="Wuv")
    Wvu = tf.Variable(tf.zeros([n_visible, n_hidden_recurrent]), name="Wvu")
    Wuu = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuu")
    bh  = tf.Variable(tf.zeros([1, n_hidden]), name="bh")
    bv  = tf.Variable(tf.zeros([1, n_visible]), name="bv")
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bu")
    u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="u0")
    BH_t = tf.Variable(tf.zeros([1, n_hidden]), name="BH_t")
    BV_t = tf.Variable(tf.zeros([1, n_visible]), name="BV_t")


    def rnn_recurrence(u_tm1, v_t):
        #Iterate through the data in the batch and generate the values of the RNN hidden nodes
        v_t  =  tf.reshape(v_t, [1, n_visible])
        u_t = tf.sigmoid(bu + tf.matmul(v_t, Wvu) + tf.matmul(u_tm1, Wuu))
        return u_t

    def visible_bias_recurrence(bv_t, u_tm1):
        #Iterate through the values of the RNN hidden nodes and generate the values of the visible bias vectors
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        return bv_t

    def hidden_bias_recurrence(bh_t, u_tm1):
        #Iterate through the values of the RNN hidden nodes and generate the values of the hidden bias vectors
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))
        return bh_t       

    #Reshape our bias matrices to be the same size as the batch.
    tf.assign(BH_t, tf.tile(BH_t, [size_bt, 1]))
    tf.assign(BV_t, tf.tile(BV_t, [size_bt, 1]))
    #Scan through the rnn and generate the value for each hidden node in the batch
    Uarr  = tf.scan(rnn_recurrence, x, initializer=u0)
    #Uarr = tf.concat([tf.reshape(u0, [1, 1, n_hidden_recurrent]), Uarr[:-1,:,:]], 0)
    #Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
    BV_t = tf.reshape(tf.scan(visible_bias_recurrence, Uarr, tf.zeros([1, n_visible])), [size_bt, n_visible])
    BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, Uarr, tf.zeros([1, n_hidden])), [size_bt, n_hidden])
    #Get the free energy cost from each of the RBMs in the batch
    cost = RBM.get_free_energy_cost(x, W, BV_t, BH_t, k=1)
    
    def reconstruction():
        #This function handles reconstructing de trajectory. This function is one of the outputs of the rnnrbm() function
        
        primer=x
        #Scan through the rnn and generate the value for each hidden node in the batch
        Uarr  = tf.scan(rnn_recurrence, primer, initializer=u0) # primer is of dimension 2, Uarr is of dimension 3
        ## without consideration of u0, give us a better result, however this change a little the configuration of rnn_rbm. actually, u impact the current bh and bv instead of future bh and bv  
        #Uarr = tf.concat([tf.reshape(u0,[1, 1, n_hidden_recurrent]), Uarr[:-1,:,:]], 0)
        #Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
        BV_t = tf.reshape(tf.scan(visible_bias_recurrence, Uarr, tf.zeros([1, n_visible])), [size_bt, n_visible])
        BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, Uarr, tf.zeros([1, n_hidden])), [size_bt, n_hidden])
        traj = RBM.gibbs_sample(primer, W, BV_t, BH_t, k=100000)
        return traj

    return x, cost, reconstruction, W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, lr, u0

