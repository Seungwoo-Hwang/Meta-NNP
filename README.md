# Meta-NNP

Branch
- main : Original NNP version
- shared_weights2 : element vector NNP version 

How to run Meta-NNP
1. Need [input.yaml], [params_ter], [run.py], [elem_list], [train_pool], [valid_pool] file
2. Command "python run.py"

Code
simple_nn/models/run.py : main structure of training procedure
simple_nn/models/loss.py : evaluate NNP energy using neural network and calculate loss function
simple_nn/models/neural_network.py : initialize neural network and initial weights
