# CEPC-AHCAL-PID
Particle Identification Using Artificial Neural Network. This net is trained on Geant4 simulation samples: electron and pion events, with energy is consistent with the particles collected during the beam test at CERN SPS&PS in 2022 and 2023.  

This project is run in Python 3.8, Pytorch-cuda=11.7. We recommend you build your virtual environment using Miniconda.
    
Train Model
        
  1. Choose the model, e.g. resnet.
     
         vi train_<model_name>.sh
     
  2. Set up your environment in the train_<model_name>.sh. You can also change other hyper-parameters.

         vi train_<model_name>.sh
         source <your_conda_env> 
         conda activate <your_env_name>
         :wq

 3. Train the model

        source train_<model_name>.sh
  
   
