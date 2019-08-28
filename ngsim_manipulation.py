'''
This is the version that take velocity as input instead of y.
for using it, make sure that the name of file is ngsim_manipulation.py
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
# UAarr到底应不应该算进u0  ***算进
# 检查高斯RBM的sigma是否正确地在应该乘除的地方添加过   ***貌似没有错误
# 高斯RBM， 尝试在重构的时候特地的把不采样，而是直接取平均值
# 采用01RBM的模型但是期望的方式 来模拟valeur reel
# 训练五个时间步长为一单位的模型
# 一个时间序列 400多步有可能太长了，试试抽样后的trajet
#  试试对轨道进行onehot之后的离散训练。 *** 完美重构
# 试试另一种正则化方法, 不求方差，只是简单的把数据压缩到01之间 然后用普通的rbm

# 改变 rnnrbm 的隐藏层的节点数  以及增加更多的可见层元素
# 目前来看，该模型只是简单的学会了产生与输入一模一样的输出，并没有 学会feature
# 添加噪声 进行监督学习
# 用relu试一下
# 可着一条traj不停学习试一下
rbm_timesteps      = 50   # timesteps in every RBM visible layer
deg_superpose=10   # degree of superposition. if deg_superpose == rbm_timesteps, it means no superposition(decalage)
class Data(object):
    def __init__(self):
        self.df = pd.read_csv("./ngsim_data/pretreatment-0750m-0805m_velocity.csv", sep=",")
        self.data=self.df.loc[:, ['Local_X','v_Vel']].values
        self.dataset = []    # a list of trajectories(normalized), every traj is a ndarray(time series)
        self.xyset = []
        i=0
        #self.ymax=np.amax(self.df.loc[:,'Local_Y'])
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
                self.xyset.append(self.df.loc[i:, ['Local_X', 'Local_Y']].values)
            else:
                traj = self.data[i:idx,:]
                traj = traj [ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
                j=0
                traj_superposed=[]
                while(j < (len(traj)-rbm_timesteps+1)  ): 
                    traj_superposed.append(  np.reshape(traj[j:j+rbm_timesteps, :],[width])   )
                    j+=deg_superpose    
                self.dataset.append(  np.array(traj_superposed)   )
                self.xyset.append(self.df.loc[i:idx-1, ['Local_X', 'Local_Y']].values)
            i = idx
        self.currentposition=0
        self.num_trajectories=len(self.dataset)
        self.max=np.loadtxt("./ngsim_data/saved_max_velocity.csv", delimiter=',')
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
            while num<idx:
                begin = begin+self.df.at[begin,'Total_Frames']
                num+=1
            end = begin+self.df.at[begin,'Total_Frames']
            traj = self.data[begin:end,:]
            traj = traj [ :(int)(np.floor(traj.shape[0]/rbm_timesteps)*rbm_timesteps) ]
            j=0
            traj_superposed=[]
            while(j < (len(traj)-rbm_timesteps+1)  ): 
                traj_superposed.append(  np.reshape(traj[j:j+rbm_timesteps, :],[width])   )
                j+=deg_superpose    
            trajectories.append( np.array(traj_superposed) )
        return trajectories
    '''
            
def pre_treatment():
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





if __name__=='__main__':
    pre_treatment()
    
    
    