This is an implementation of RNN-RBM for NGSIM for learning distribution of vehicle trajectories.

First, you need to put the "trajectories-0750am-0805am.csv"  in the directory ./ngsim_data/
1. python ngsim_manipulation.py  # pretreatement of dataset

2. python weight_initialization.py  # initialization of W of RBM

3. python rnn_rbm_train.py  <num_epoch> # training,  num_epoch can be about 20

4.python rnn_rbm_reconstruction.py ./parameter_checkpoints/<fichier .ckpt>  # reconstruction