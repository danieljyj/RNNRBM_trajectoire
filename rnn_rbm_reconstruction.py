import tensorflow as tf
import sys
import os
from tqdm import tqdm
import rnn_rbm
import ngsim_manipulation
import draw
import numpy as np

"""
    This file contains the code for running a tensorflow session to reconstruct trajectories
"""

# 三条比较有代表性的traj 分别是    get_traj(1)     get_traj(7)   get_traj(9) 其中第九条偏差最多

num_sample=1

def main(saved_weights_path):
    print("reading data...")
    data=ngsim_manipulation.Data()
    #This function takes as input the path to the weights of the network
    x, cost, reconstruction, W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, lr, u0 = rnn_rbm.rnnrbm()#First we build and get the parameters of the network

    params=[W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    saver = tf.train.Saver(params) #We use this saver object to restore the weights of the model
    
    #trajectories_primer = data.get_trajectories(start=0, end=5)
    #trajectories_primer = data.get_trajs([0,3,4])
    trajectories_primer = [data.get_traj(1)]
    
    # add noise to trajectories
    trajectories_primer = add_noise(trajectories_primer, data.max)
    
    reconstructed_trajectories=[]
    decontr_trajectories_primer=[]
    print("reconstruction...")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, saved_weights_path) #load the saved weights of the network
        for j in tqdm( range(len(trajectories_primer)) ):
            # 如果是disData, 用下面这行
            # reconstructed_trajectories.append( sess.run(reconstruction(), feed_dict={x: trajectories_primer[j]}) )
            for i in range(num_sample):
                decontract_traj = data.decontraction(   sess.run(reconstruction(), feed_dict={x: trajectories_primer[j]})   )
                reconstructed_trajectories.append( decontract_traj ) #Prime the network with primer and reconstruct this trajectory
            decontr_trajectories_primer.append(  data.decontraction(trajectories_primer[j])    )

        trajectories = decontr_trajectories_primer + reconstructed_trajectories

        draw.draw_trajectories(trajectories, num_trajs = len(trajectories_primer)  )
def add_noise(trajectories, max):
    for i in range(len(trajectories)):
        print(type(trajectories))
        print(type(trajectories[i][10:20]))
        print(trajectories[i][10:20])
        trajectories[i][10:20] = trajectories[i][10:20]+[10,0]/max
    return trajectories



if __name__ == "__main__":
    main(sys.argv[1])
    
