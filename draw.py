import numpy as np
import matplotlib.pyplot as plt
import ngsim_manipulation
import os


def draw_trajectories(trajectories, num_trajs):
    if not os.path.isdir("./picture_folder"):
        os.makedirs("./picture_folder")
    plt.figure(figsize=(30,15))
    idx=0
    for traj in trajectories[: num_trajs]:
        plt.plot(traj[:,0], traj[:,1],'r',label='original.{}'.format(idx))
        idx+=1
    idx=0
    for traj in trajectories[num_trajs:]:
        plt.plot(traj[:,0], traj[:,1], label='reconstructed.{}'.format(idx))
        idx+=1
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("trajectories")
    plt.legend()
    plt.savefig("./picture_folder/reconstructed trajectories")
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

    
def draw_traj(traj):
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
    