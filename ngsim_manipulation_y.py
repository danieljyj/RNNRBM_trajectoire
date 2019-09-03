'''
This is the version that take y as input instead of velocity.
contraction dataset to [0,1]
for using it, make sure that the name of file is ngsim_manipulation.py
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

rbm_timesteps      = 10   # timesteps in every RBM visible layer
deg_superpose=5   # degree of superposition. if deg_superpose == rbm_timesteps, it means no superposition
class Data(object):
    def __init__(self):
        df = pd.read_csv("./ngsim_data/pretreatment-0750m-0805m.csv", sep=",")
        self.data=df.loc[:, ['Local_X','Local_Y']].values
        self.dataset=[]    # a list of trajectories(normalized), every traj is a ndarray(time series)
        i=0
        width = (int)(self.data.shape[1]*rbm_timesteps)
        while( i < len(self.data)):
            idx=i+df.at[i,'Total_Frames']
            if idx > len(self.data):
                traj = self.data[i:,:]
                traj = traj [ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
                j=0
                traj_superposed=[]
                while(j < (len(traj)-rbm_timesteps+1)  ): 
                    traj_superposed.append(  np.reshape(traj[j:j+rbm_timesteps, :],[width])   )
                    j+=deg_superpose    
                self.dataset.append(   np.array( traj_superposed)   )
                #traj = traj [ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
                #self.dataset.append(   np.reshape(traj, [(int)(len(traj)/rbm_timesteps), width])    )
                #self.dataset.append(   self.data[i::5,:]   )
            else:
                traj = self.data[i:idx,:]
                traj = traj [ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
                j=0
                traj_superposed=[]
                while(j < (len(traj)-rbm_timesteps+1)  ): 
                    traj_superposed.append(  np.reshape(traj[j:j+rbm_timesteps, :],[width])   )
                    j+=deg_superpose    ##1 is the degree of superpose. if j += rbm_timesteps, it means no superpose
                self.dataset.append(  np.array(traj_superposed)   )
                #traj = traj [ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
                #self.dataset.append(    np.reshape(traj, [(int)(len(traj)/rbm_timesteps), width])     )
                #self.dataset.append( self.data[i:idx:5,:]   )
                #assert(df.at[i,'Vehicle_ID'] != df.at[idx,'Vehicle_ID'])
            i = idx
        self.currentposition=0
        self.num_trajectories=len(self.dataset)
        self.max=np.loadtxt("./ngsim_data/saved_max.csv", delimiter=',')
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!shape of one trajectoire is : ", self.dataset[0].shape)
    def next_traj(self):
        if self.currentposition == self.num_trajectories:
            raise Exception('End of dataset')
        traj = self.dataset[self.currentposition]
        self.currentposition += 1
        if self.currentposition == self.num_trajectories :
            self.currentposition=0
        return traj
    def get_traj(self, numero):
        if numero >= self.num_trajectories:
            raise Exception("numero is bigger than total numbers")
        traj=self.dataset[numero]
        return traj
    def get_trajs(self, numeros):
        trajs=[]
        for numero in numeros:
            trajs.append(self.dataset[numero])
        return trajs
    def get_trajectories(self, start, end):
        return self.dataset[start: end]

    def decontraction(self, traj):
        decontract = traj.copy()
        decontract = decontract*np.tile(self.max , rbm_timesteps)
        return decontract
    '''
    def denormalisation(self, traj):
        denorm=traj.copy()
        for j in range(len(std)):
            if self.std[j]==0:
                raise Exception("std[{}] is 0".format(j))
        denorm=denorm*self.std
        denorm += self.mean
        return denorm
    '''
class Dis_Data(object):
    def __init__(self):
        df = pd.read_csv("./ngsim_data/trajectories-0750am-0805am.csv", sep=",")
        df_lane = df.loc[:, 'Lane_ID']
        df_lane=pd.get_dummies(df_lane)
        self.data=df_lane.values
        np.savetxt("./output_folder/data.csv", self.data)

        self.dataset=[]    # a list of trajectories(normalized), every traj is a ndarray(time series)
        i=0
        while( i < len(self.data)):
            idx=i+df.at[i,'Total_Frames']
            if idx > len(self.data):
                self.dataset.append(self.data[i::5,:])
            else:
                self.dataset.append(self.data[i:idx:5,:])
                assert(df.at[i,'Vehicle_ID'] != df.at[idx,'Vehicle_ID'])
            i = idx
        self.currentposition=0
        self.num_trajectories=len(self.dataset)
    def next_traj(self):
        if self.currentposition == self.num_trajectories:
            raise Exception('End of dataset')
        traj = self.dataset[self.currentposition]
        self.currentposition += 1
        if self.currentposition == self.num_trajectories :
            self.currentposition=0
        return traj
    def get_traj(self, numero):
        if numero >= self.num_trajectories:
            raise Exception("numero is bigger than total numbers")
        traj=self.dataset[numero]
        return traj
    def get_trajectories(self, num):
        return self.dataset[:num]
    
def onehot_to_category(onehots):
    catg = [np.argmax(onehot) for onehot in onehots]
    return catg
    
def write_traj(traj):
    if not os.path.isdir("./output_folder"):
        os.makedirs("./output_folder")
    df = pd.DataFrame(traj)
    df.to_csv("./output_folder/reconstructed_trajectory.csv", sep=",", header=False)
    
def pre_treatment():
    if not os.path.isdir("./ngsim_data"):
        os.makedirs("./ngsim_data")
    df = pd.read_csv("./ngsim_data/trajectories-0750am-0805am.csv", sep=",")
    data=df.loc[:, ['Local_X','Local_Y']].values
    def contraction(data):
        max=np.amax(data, axis=0)
        data=data/max
        np.savetxt("./ngsim_data/saved_max.csv", max, delimiter=",")
        return data
    data=contraction(data)
    df.loc[:, ['Local_X','Local_Y']] = data
    df.to_csv("./ngsim_data/pretreatment-0750m-0805m.csv", sep=',', index=False)





if __name__=='__main__':
    pre_treatment()
    
    
    