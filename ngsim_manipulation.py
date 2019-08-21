import numpy as np
import pandas as pd
from tqdm import tqdm
import os
#UAarr到底应不应该算进u0
# 检查高斯RBM的sigma是否正确地在应该乘除的地方添加过   ***貌似没有错误
# 高斯RBM， 尝试在重构的时候特地的把不采样，而是直接取平均值
# 采用01RBM的模型但是期望的方式 来模拟valeur reel
# 训练五个时间步长为一单位的模型
# 一个时间序列 400多步有可能太长了，试试抽样后的trajet
#  试试对轨道进行onehot之后的离散训练。 *** 完美重构
# 试试另一种正则化方法, 不求方差，只是简单的把数据压缩到01之间 然后用普通的rbm

# 改变 rnnrbm 的隐藏层的节点数  以及增加更多的可见曾元素
# 目前来看，该模型只是简单的学会了产生与输入一模一样的输出，并没有 学会feature

rbm_timesteps      = 1   # timesteps in every RBM visible layer
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
                #self.dataset.append(   np.reshape(self.data[i::5,:], [(int)((len(self.data)-i)/rbm_timesteps), width])    )
                self.dataset.append(   self.data[i::5,:]   )
            else:
                #self.dataset.append(    np.reshape(self.data[i:idx:5,:], [(int)((idx-i)/rbm_timesteps), width])     )
                self.dataset.append( self.data[i:idx:5,:]   )
                assert(df.at[i,'Vehicle_ID'] != df.at[idx,'Vehicle_ID'])
            i = idx
        self.currentposition=0
        self.num_trajectories=len(self.dataset)
        self.max=np.loadtxt("./ngsim_data/saved_max.csv", delimiter=',')
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
        decontract = decontract*self.max
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
    
    
    