'''
This is the version that draw x and velocity
'''
import numpy as np
import matplotlib.pyplot as plt
import ngsim_manipulation
import os

rbm_timesteps = ngsim_manipulation.rbm_timesteps
deg_superpose = ngsim_manipulation.deg_superpose
def draw_trajectories(inpt_trajectories, num_trajs):
    trajectories=[]
    for traj in inpt_trajectories[:]:
        j=0
        traj_sequence=[]
        for unit in traj:
            traj_sequence.append( np.reshape(unit[: (int)(len(unit)/rbm_timesteps*deg_superpose) ], [deg_superpose, -1])    )
        traj = np.reshape(np.array(traj_sequence),[-1, (int)(len(unit)/rbm_timesteps)])
        trajectories.append(traj)
#    trajectories=trajectories+ inpt_trajectories[-num_trajs:]
    if not os.path.isdir("./picture_folder"):
        os.makedirs("./picture_folder")
    plt.figure(figsize=(30,15))
    idx=0
    for traj in trajectories[: num_trajs]:
        plt.plot(traj[:,0], 0.1*np.cumsum(traj[:,1]),'r',label='original.{}'.format(idx))
        idx+=1
    idx=0
    for traj in trajectories[num_trajs:]:
        plt.plot(traj[:,0], 0.1*np.cumsum(traj[:,1]), label='reconstructed.{}'.format(idx))
        idx+=1
    '''
    idx=0
    for traj in trajectories[(-num_trajs): ]:
        print("type,", type(traj))
        print("shape,", traj.shape)
        plt.plot(traj[:,0], traj[:,1], label='xy.{}'.format(idx))
        idx+=1
    '''
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("trajectories")
    plt.legend()
    plt.savefig("./picture_folder/reconstructed trajectories")
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

    
def draw_traj(traj):
    traj = np.reshape(traj, [(int)(len(traj)*rbm_timesteps), -1])
    if not os.path.isdir("./picture_folder"):
        os.makedirs("./picture_folder")
    plt.figure(figsize=(30,15))
    plt.plot(traj[:,0], traj[:,1], label='vehicle')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("trajectories")
    plt.legend()
    plt.savefig("./picture_folder/reconstructed traj")
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
def draw_dis_traj(traj):
    if not os.path.isdir("./picture_folder"):
        os.makedirs("./picture_folder")
    con_traj=ngsim_manipulation.onehot_to_category(traj)
    y = 20*np.arange(1. ,  (len(con_traj)+1))
    plt.figure(figsize=(30,15))
    plt.plot(con_traj, y, label='vehicle')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("trajectories")
    plt.legend()
    plt.savefig("./picture_folder/reconstructed dis_traj")
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def draw_dis_trajectories(trajectories):
    if not os.path.isdir("./picture_folder"):
        os.makedirs("./picture_folder")
    plt.figure(figsize=(30,15))
    for traj in trajectories:
        x=ngsim_manipulation.onehot_to_category(traj)
        y = 20*np.arange(1. ,  (len(x)+1))
        plt.plot(x, y, label='vehicle')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("trajectories")
    plt.legend()
    plt.savefig("./picture_folder/reconstructed dis_trajectories")
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    
def draw_all_trajectories():
    if not os.path.isdir("./picture_folder"):
        os.makedirs("./picture_folder")
    data =ngsim_manipulation.Data()
    trajectories = data.dataset
    plt.figure(figsize=(30,15))
    for traj in trajectories:
        traj = data.decontraction(traj) 
        plt.plot(traj[:,0], traj[:,1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("All trajectories")
    plt.legend()
    plt.savefig("./picture_folder/All_trajectories")
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

    
if __name__ == '__main__':
    draw_all_trajectories()
    