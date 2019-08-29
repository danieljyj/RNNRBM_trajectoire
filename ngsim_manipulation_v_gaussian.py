'''
This is the version that take velocity as input instead of y. 
normalisation to gaussian distribution.
for using it, make sure that the name of file is ngsim_manipulation.py
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


rbm_timesteps      = 50   # timesteps in every RBM visible layer
deg_superpose=10   # degree of superposition. if deg_superpose == rbm_timesteps, it means no superposition(decalage)
class Data(object):
    def __init__(self):
        self.df = pd.read_csv("./ngsim_data/pretreatment-0750m-0805m_velocity_gaussian.csv", sep=",")
        self.data=self.df.loc[:, ['Local_X','v_Vel']].values
        self.dataset = []    # a list of trajectories(normalized), every traj is a ndarray(time series)
        i=0
        width = (int)(self.data.shape[1]*rbm_timesteps)
        while( i < len(self.data)):
            idx=i+self.df.at[i,'Total_Frames']
            if idx > len(self.data):
                traj = self.data[i:,:]
                traj = traj[ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
                j=0
                traj_superposed=[]
                while(j < (len(traj)-rbm_timesteps+1)  ): 
                    traj_superposed.append(  np.reshape(traj[j:j+rbm_timesteps, :],[width])   )
                    j+=deg_superpose    
                self.dataset.append(   np.array( traj_superposed)   )
            else:
                traj = self.data[i:idx,:]
                traj = traj [ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
                j=0
                traj_superposed=[]
                while(j < (len(traj)-rbm_timesteps+1)  ): 
                    traj_superposed.append(  np.reshape(traj[j:j+rbm_timesteps, :],[width])   )
                    j+=deg_superpose    
                self.dataset.append(  np.array(traj_superposed)   )
            i = idx
        self.currentposition=0
        self.num_trajectories=len(self.dataset)
        self.mean_std=np.loadtxt("./ngsim_data/saved_mean_std_velocity.csv", delimiter=',')
    def next_traj(self):
        if self.currentposition == self.num_trajectories:
            raise Exception('End of dataset')
        traj = self.dataset[self.currentposition]
        self.currentposition += 1
        if self.currentposition == self.num_trajectories :
            self.currentposition=0
        return np.copy(traj)
    def get_traj(self, numero):
        if numero >= self.num_trajectories:
            raise Exception("numero is bigger than total numbers")
        traj=self.dataset[numero]
        return np.copy(traj)
    def get_trajs(self, numeros):
        trajs=[]
        for numero in numeros:
            trajs.append(np.copy(self.dataset[numero]))
        return trajs
    def get_trajectories(self, start, end):
        return np.copy(self.dataset[start: end])

    def denormalisation(self, traj):
        denorm=traj.copy()
        denorm=denorm*np.tile(self.mean_std[1] , rbm_timesteps)
        denorm += np.tile(self.mean_std[0],rbm_timesteps)
        return denorm
    
    def add_noise_gaussian(self, trajectories):
        idx= [i*2 for i in range(rbm_timesteps)]
        for i in range(len(trajectories)):
            trajectories[i][:,idx] += 2*np.random.randn(len(trajectories[i]), rbm_timesteps)/self.max[0]
        return trajectories
    '''
    def add_noise_zero(self, idx):
        trajectories=[]
        begin=0
        num=0
        width = (int)(self.data.shape[1]*rbm_timesteps)
        for i in idx:
            while num<i:
                begin = begin+self.df.at[begin,'Total_Frames']
                num+=1
            end = begin+self.df.at[begin,'Total_Frames']
            traj = self.data[begin:end,:]
            traj = traj [ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
            traj[100:150,0]=0
            j=0
            traj_superposed=[]
            while(j < (len(traj)-rbm_timesteps+1)  ): 
                traj_superposed.append(  np.reshape(traj[j:j+rbm_timesteps, :],[width])   )
                j+=deg_superpose    
            trajectories.append( np.array(traj_superposed) )
        return trajectories
    '''
'''
# function that contracts data in [0,1]
def pre_treatment_contraction():
    if not os.path.isdir("./ngsim_data"):
        os.makedirs("./ngsim_data")
    df = pd.read_csv("./ngsim_data/trajectories-0750am-0805am.csv", sep=",")
    data=df.loc[:, ['Local_X', 'v_Vel']].values
    def contraction(data):
        max=np.amax(data, axis=0)
        data=data/max
        np.savetxt("./ngsim_data/saved_max_velocity.csv", max, delimiter=",")
        return data
    data=contraction(data)
    df.loc[:, ['Local_X', 'v_Vel']] = data
    df.to_csv("./ngsim_data/pretreatment-0750m-0805m_velocity.csv", sep=',', index=False)
'''
#function that makes data follow gaussian distribution  N(0,1)
def pre_treatment_gaussian():
    if not os.path.isdir("./ngsim_data"):
        os.makedirs("./ngsim_data")
    df = pd.read_csv("./ngsim_data/trajectories-0750am-0805am.csv", sep=",")
    data=df.loc[:, ['Local_X', 'v_Vel']].values
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    mean_std=np.array([mean, std])
    np.savetxt("./ngsim_data/saved_mean_std_velocity.csv", mean_std, delimiter=",")
    data = data-mean
    data=data/std
    df.loc[:, ['Local_X', 'v_Vel']] = data
    df.to_csv("./ngsim_data/pretreatment-0750m-0805m_velocity_gaussian.csv", sep=',', index=False)




if __name__=='__main__':
    pre_treatment_gaussian()
    
    
    