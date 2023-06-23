# RLFold: Reinforcement learning environment for RNA 3D Structure Prediction

##### Reinforcement Learning for RNA 3D Structure Prediction #####

This project aims to predict the 3D structure of RNA sequences with reinforcement learning. The approach is based on developing a model/agent trained with the available 3D structures in Protein Data Bank. This model/agent could then be used for prediction.

The agent is trained with RNA sequences from PDB that are up to 20 nucleotides long. Relatively short sequences are considered for now for the sake of simplicity.

The training procedure is based on folding the structures with RMSD as a reward function. Starting with an initial structure, the RMSD between the predicted and target structure is used to build up a policy.

This trained policy is then used to tackle sequences of unknown structure. As an initial start, already-available prediction tools could be used to get a rough estimate of the sequence.

The model takes as input the torsion angle readings in addition to the sequence encoding. The output from the model are the perturbation amounts that are going to be applied to the current structure.

##### Environment Setup #####

### Create Conda environment

$ conda env create -f environment.yml

### Activate Conda environment

$ conda activate myenv

### The database/ folder contains the training dataset, which is composed of 260 structures, and the test dataset, which contains a single structure.

### Run the code

$ python RLFold/main.py

### At the end of training, results will be available as episode_rmsd.png and final_rmsd.png:

	## episode_rmsd.png shows how RMSD of the test structure changes with respect to its native structure during an episode.

	## final_rmsd.png shows the final RMSD achieved at the end of an episode.

#### Training timesteps: 1.000.000

#### Test interval: 10.000

### Maximum sequence length to be used in dataset: 20 nucleotides

### policy.zip is the model saved at the end of training.
