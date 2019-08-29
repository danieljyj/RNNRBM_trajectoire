'''
Bernoulli-Bernoulli RBM, but with visible layer expectation value instead of sampling
'''
import tensorflow as tf
import numpy as np

def sample(probs):
    #returns vector of 0 and 1
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), minval=0, maxval=1))

#This function runs the gibbs chain.
def gibbs_sample(x, W, bv, bh, k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv) #Propagate the hidden values to sample the visible values
        #xk = tf.nn.relu(tf.matmul(hk, tf.transpose(W)) + bv)  # in Hinton's article, he said that Rectified unit perform well, I didn't observe it.
        return count+1, k, xk

    #Run gibbs steps for k iterations
    ct = tf.constant(0) # counter
    [_, _, x_sample] = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x])
    #We need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample
    
def F(x, W, bv, bh):
    #The function computes the free energy of a visible vector. 
    return -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(x, W) + bh)), axis=1) - tf.reduce_sum(tf.multiply(x, bv), axis=1)

def get_free_energy_cost(x, W, bv, bh, k):
    #We use this function in training to get the free energy cost of the RBM. We can pass this cost directly into TensorFlow's optimizers 
    #First, draw a sample from the RBM
    x_sample   = gibbs_sample(x, W, bv, bh, k)

    #The cost is based on the difference in free energy between x and xsample
    cost = tf.reduce_mean(tf.subtract(F(x, W, bv, bh ), F(x_sample, W, bv, bh)))  #+ 10*tf.norm(W, ord=2) # useless for avoiding over-fitting.
    return cost

def get_cd_update(x, W, bv, bh, k, lr):
    #This is the contrastive divergence algorithm. Used just for weight initialization.

    x_sample = gibbs_sample(x, W, bv, bh, k)
    # probability of hidden layer
    ph = tf.sigmoid(tf.matmul(x, W) + bh)
    #probability of visible layer
    ph_sample = tf.sigmoid(tf.matmul(x_sample, W) + bh)

    #Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
    lr = tf.constant(lr, tf.float32) #The CD learning rate
    size_bt = tf.cast(tf.shape(x)[0], tf.float32) #The batch size
    W_  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(tf.transpose(x), ph), tf.matmul(tf.transpose(x_sample), ph_sample))) # here we use probability instead of sample of hidden layer
    bv_ = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
    bh_ = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(ph, ph_sample), 0, True))  # here we use probability instead of sample of hidden layer

    #When we do sess.run(updt), TensorFlow will run all 3 update steps
    updt = [W.assign_add(W_), bv.assign_add(bv_), bh.assign_add(bh_)]
    return updt

