'''
Gaussian Bernoulli RBM
'''
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
from math import isnan

sigma=1.
#This function lets us easily sample from a vector of probabilities
def sample_bernouille(probs):
    #returns vector of 0 and 1
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), minval=0, maxval=1))
def sample_gaussian(mean , sigma):
    #return vector of real number, according to gaussian ditribution
    return tf.random.normal(tf.shape(mean), mean=mean, stddev=sigma)
    
#This function runs the gibbs chain.
def gibbs_sample(x, W, bv, bh, k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        hk = sample_bernouille(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample_gaussian( tf.matmul(hk*sigma, tf.transpose(W))+bv, sigma) #Propagate the hidden values to sample the visible values
        #xk = tf.nn.relu(xk)
        return count+1, k, xk

    #Run gibbs steps for k iterations
    ct = tf.constant(0) #counter
    [_, _, x_sample] = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x])
    #We need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

def F(x, W, bv, bh):
    #The function computes the free energy of a visible vector.
    hidden_term = tf.reduce_sum(tf.log(1. + tf.exp(tf.matmul(x/sigma, W) + bh)), axis=1)
    vbias_term = tf.reduce_sum(tf.square(tf.subtract(x, bv)), axis=1)/((sigma**2)*2)
    return -hidden_term + vbias_term

def get_free_energy_cost(x, W, bv, bh, k):
    #We use this function in training to get the free energy cost of the RBM. We can pass this cost directly into TensorFlow's optimizers 
    #First, draw a sample from the RBM
    x_sample   = gibbs_sample(x, W, bv, bh, k)
    #The cost is based on the difference in free energy between x and xsample
    cost = tf.reduce_mean(tf.subtract(F(x, W, bv, bh ), F(x_sample, W, bv, bh))) # + tf.norm(W, ord=2) #没用， 起不到抑制overfitting的作用
    return cost
    
# by explicit calculation 
def get_cd_update(x, W, bv, bh, k, lr):
    #This is the contrastive divergence algorithm. Used just for weight initialization.

    x_sample = gibbs_sample(x, W, bv, bh, k)
    #The sample of the hidden nodes, starting from the visible state of x
    ph = tf.sigmoid(tf.matmul(x, W) + bh)
    #The sample of the hidden nodes, starting from the visible state of x_sample
    ph_sample = tf.sigmoid(tf.matmul(x_sample, W) + bh)

    #Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
    lr = tf.constant(lr, tf.float32) #The CD learning rate
    size_bt = tf.cast(tf.shape(x)[0], tf.float32) #The batch size
    W_  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(tf.transpose(x), ph), tf.matmul(tf.transpose(x_sample), ph_sample)))
    W_= tf.divide(W_,sigma)
    bv_ = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), axis=0,keepdims=True))
    bv_=tf.divide(bv_, sigma**2)
    bh_ = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(ph, ph_sample), axis=0, keepdims=True))

    #When we do sess.run(updt), TensorFlow will run all 3 update steps
    updt = [W.assign_add(W_), bv.assign_add(bv_), bh.assign_add(bh_)]
    return updt
'''
# by gradient function
def get_cd_update(x, W, bv, bh, k, lr):
    x_sample = gibbs_sample(x, W, bv, bh, k)
    # cost function;
    cost = get_free_energy_cost(x, W, bv, bh, k)
    params = [W, bh, bv]
    opt_func = tf.train.AdamOptimizer(learning_rate = lr) 
    grad_and_params = opt_func.compute_gradients(cost, params)
    grad_and_params = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grad_and_params]
    updt = opt_func.apply_gradients(grad_and_params)
    return updt 
'''