import tensorflow as tf
import sys
from tqdm import tqdm
import rnn_rbm
import ngsim_manipulation
import draw
import numpy as np
import RBM

"""
    This file contains the code for running a tensorflow session to reconstruct trajectories
"""


rbm_timesteps=ngsim_manipulation.rbm_timesteps
num_sample=1  # number of sampling

#This function takes as input the path to the weights of the network
def main(saved_weights_path):
    print("reading data...")
    data=ngsim_manipulation.Data()

    x, cost, reconstruction, W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, lr, u0 = rnn_rbm.rnnrbm()  # First we build and get the parameters of the network
    params=[W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    saver = tf.train.Saver(params) #We use this saver object to restore the weights of the model
    
    #get many trajectories
    #trajectories_primer = data.get_trajectories(start=0, end=5)
    idx=[0,3,4]    # three representative traj:   get_traj(0)     get_traj(3)   get_traj(4) 
    trajectories_primer = data.get_trajs(idx)
    trajectories_primer = data.add_noise_gaussian(trajectories_primer)
    #trajectories_primer = data.add_noise_zero(idx)
    
    #get one traj
    #idx=0
    #trajectories_primer = [data.get_traj(idx)]
    #trajectories_primer = data.add_noise_gaussian(trajectories_primer)
    #trajectories_primer = data.add_noise_zero([idx])

    #xytraj=data.xyset[1]
    #xytraj = xytraj*[data.max[0],1]
    
    #x_sample=RBM.gibbs_sample(x, W, bv, bh, 1)
    reconstructed_trajectories=[]
    decontr_trajectories_primer=[]
    print("reconstruction...")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, saved_weights_path) #load the saved weights of the network
        '''
        # without time dependence, reconstruction based on a single RBM(only one RBM for all time steps) 
        # actually, this performs not too badly, even just one RBM for different time(because we choose rbm_steps=100m it include time dependence already) 
        for j in tqdm( range(len(trajectories_primer)) ):
            decontract_traj = data.decontraction( sess.run(x_sample, feed_dict={x: trajectories_primer[j]})  )
            reconstructed_trajectories.append( decontract_traj ) 
            decontr_trajectories_primer.append(  data.decontraction(trajectories_primer[j])    )
        '''
        
        # reconstruction based on RNN-RBM, time dependance. Expectation RBM
        for j in tqdm( range(len(trajectories_primer)) ):
            for i in range(num_sample):
                decontract_traj = data.decontraction(   sess.run(reconstruction(), feed_dict={x: trajectories_primer[j]})   )
                reconstructed_trajectories.append( decontract_traj ) #Prime the network with primer and reconstruct this trajectory
            decontr_trajectories_primer.append(  data.decontraction(trajectories_primer[j])    )
        
        trajectories = decontr_trajectories_primer + reconstructed_trajectories #+ [xytraj]
        draw.draw_trajectories(trajectories, num_trajs = len(trajectories_primer)  )
        '''
        # reconstruction based on RNN-RBM, time dependance. GBRBM
        for j in tqdm( range(len(trajectories_primer)) ):
            for i in range(num_sample):
                denorm_traj = data.denormalisation(   sess.run(reconstruction(), feed_dict={x: trajectories_primer[j]})   )
                reconstructed_trajectories.append( denorm_traj ) #Prime the network with primer and reconstruct this trajectory
            decontr_trajectories_primer.append(  data.denormalisation(trajectories_primer[j])    )
        
        trajectories = decontr_trajectories_primer + reconstructed_trajectories 
        draw.draw_trajectories(trajectories, num_trajs = len(trajectories_primer)  )
        '''


if __name__ == "__main__":
    main(sys.argv[1])
    
