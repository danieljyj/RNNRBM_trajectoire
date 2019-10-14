# Unsupervised Vehicle Trajectory learning in multi-lane highway
by [Yingjie Jiang](https://www.yingjiejiang.com)

This is an adapted variaiton of [RNN-RBM](https://arxiv.org/ftp/arxiv/papers/1206/1206.6392.pdf) for unsupervised vehicle trajectory pattern learning on a multi-lane highway. The dataset used is [US Highway 101](https://www.fhwa.dot.gov/publications/research/operations/07030/).

Unlike the original RNN-RBM, this is a model which performs well for real-valued dataset. After training, it can memorize a given vehicle trajectories pattern on a specific route, which can be used to reconstruct vehicle trajectories based on a input dataset polluted by noise.

First, you need to put the "trajectories-0750am-0805am.csv"  in the directory ./ngsim_data/
1. `python ngsim_manipulation.py`  # pretreatement of dataset

2. `python weight_initialization.py`  # initialization of W of RBM

3. `python rnn_rbm_train.py  <num_epoch>` # training,  num_epoch can be about 10

4. `python rnn_rbm_reconstruction.py ./parameter_checkpoints/<fichier .ckpt>`  # reconstruction

5. The output picture is stocked in picture_folder.

There are different versions of ngsim_manipulation, rnn_rbm, RBM, draw, they treat different situations. For using them, rename them(delete the version notation).
