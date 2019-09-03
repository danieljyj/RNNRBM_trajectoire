"""
This is the second version of rnn_rbm, in which we generate every u_t based on the sampled result of v_t, 
whereas in the first version, we just sample all the u_t in the same time based on the original input vector of v_t(e.g. x)
if you want to use this version , rename the file to rnn_rbm.py
"""
import tensorflow as tf
import numpy as np
import RBM
import ngsim_manipulation

n_visible          = 2*ngsim_manipulation.rbm_timesteps #The size of the RBM visible layer
n_hidden           = (int)(n_visible/2)  #The size of the RBM hidden layer
n_hidden_recurrent = 50 #The size of each RNN hidden layer

def rnnrbm():

    #This function builds the RNN-RBM and returns the parameters of the model
    x  = tf.placeholder(tf.float32, [None, n_visible]) #The placeholder variable that holds our data
    lr  = tf.placeholder(tf.float32) #The learning rate. We set and change this value during training.
    
    size_bt = tf.shape(x)[0] # Not really batch, but the longeur of time series

    #parameters to learn, we find that except W, the other four Weight matrices aren't well learned,
    #So there isn't a good evolution of RBM in time. We can train a single RBM and produce a similar result.
    W   = tf.Variable(tf.zeros([n_visible, n_hidden]), name="W")
    Wuh = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden]), name="Wuh")
    Wuv = tf.Variable(tf.zeros([n_hidden_recurrent, n_visible]), name="Wuv")
    Wvu = tf.Variable(tf.zeros([n_visible, n_hidden_recurrent]), name="Wvu")
    Wuu = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuu")
    bh  = tf.Variable(tf.zeros([1, n_hidden]), name="bh")     # bh is altually bh_0, we consider only bh_0 as parameters for training .
    bv  = tf.Variable(tf.zeros([1, n_visible]), name="bv")    
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bu")
    u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="u0")
    BH_t = tf.Variable(tf.zeros([1, n_hidden]), name="BH_t")
    BV_t = tf.Variable(tf.zeros([1, n_visible]), name="BV_t")


    def recurrence(nest, v_t):
    # This function do all the procedures in one RBM.
        bv_t = tf.add(bv, tf.matmul(nest[0], Wuv)) # generate current bv_t based on u_{t-1}
        bh_t = tf.add(bh, tf.matmul(nest[0], Wuh)) # generate current bh_t based on u_{t-1}
        v_t_sample = RBM.gibbs_sample(tf.reshape(v_t, [1, n_visible]),  W,  bv_t , bh_t, k=1 ) # sample v_t according generated bias
        v_t_sample  =  tf.reshape(v_t_sample, [1, n_visible])  
        u_t = tf.sigmoid(bu + tf.matmul(v_t_sample, Wvu) + tf.matmul(nest[0], Wuu))  # calculate current u_t
        
        return [u_t, bv_t, bh_t, v_t_sample]
        
    Uarr, BV_t, BH_t, V_t_sample = tf.scan(recurrence, x, initializer = [u0, tf.zeros([1, n_visible]), tf.zeros([1, n_hidden]), tf.zeros([1, n_visible])]     )
    BV_t = tf.reshape(BV_t, [size_bt, n_visible])
    BH_t = tf.reshape(BH_t, [size_bt, n_hidden])
    V_t_sample = tf.reshape(V_t_sample, [size_bt, n_visible])
    #Get the free energy cost from each of the RBMs 
    cost = tf.reduce_mean(tf.subtract(RBM.F(x, W, BV_t, BH_t ), RBM.F(V_t_sample, W, BV_t, BH_t)))
    
    def reconstruction():
        #This function handles reconstructing de trajectory. This function is one of the outputs of the rnnrbm() function
        primer=x
        Uarr, BV_t, BH_t, V_t_sample = tf.scan(recurrence, x, initializer = [u0, bv, bh, tf.zeros([1, n_visible])]     )
        traj = tf.reshape(V_t_sample, [size_bt, n_visible])
        return traj

    return x, cost, reconstruction, W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, lr, u0

