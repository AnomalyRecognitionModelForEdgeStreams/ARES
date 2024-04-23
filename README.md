# ARES

### REQUIREMENTS
To run ARES you will be asked to install the packages listed in requirements.txt. 
Python 3.10.12 has been used.

### EXPERIMENTS
Command for running ARES: python main.py config/configuration.json

In particular, in configuration.json you need to fill the following specifications:

#### General:

1) dataset: "dataset_name"
2) dataset_extension: ".csv"
3) model_name: "GAE"
4) GPU: 0
5) seed: 42

#### Data:
1) save_files: 0/1 -- 1 for processing a new dataset from data/raw, 0 for loading a processed dataset from data/processed
2) perc_train: 0.6 -- percentage of data for training
3) perc_val: 0.2 -- percentage of data for validation

#### Methodology:
1) training: 0/1 -- training the GAE
2) validation: 0/1 -- running the tuning of the hyper-parameters
3) update_hst: 0/1 -- running ARES-Static/ARES-Dynamic
4) phi: "mean"/"minus" -- Eq. 1/2 
5) random_start: 0/1 -- HST initialization from training data: 0 -> last window, 1 -> uniform sampling. 

#### Training:
1) in_channels: 32
2) hidden_channels: 16
3) out_channels: 8
4) num_layers: 3
5) epochs: 10000
6) learning_rate: 0.001
7) patience: 1000
8) plot_title: "plot_loss/plot_loss_GAE_dataset_name"
9) model_save_path: "checkpoints/model_GAE_dataset_name"

#### Test:
1) n_trees: 8
2) height: 6
3) window_size: 64
4) thresholds: "0.859, 0.739, 0.876, 0.854, 0.840, 0.865, 0.901, 0.855, 0.875",
5) weights: "1.0,0.0,0.0" -- w_1, w_2, and w_3 from Eq 4
6) window_already_seen: 64 -- Number of steps before resetting the cache memory
  
### Note:
1) Configuration files for the dataset used are in the "config" folder.
2) data/processed/IDS2018 and data/processed/DARPA need to be uncompressed due to the size dimension limit on github.
