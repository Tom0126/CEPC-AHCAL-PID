# CEPC-AHCAL-PID
Particle Identification Using Artificial Neural Network. This net is trained on Geant4 simulation samples: electron and pion events, with energy is consistent with the particles collected during the beam test at CERN SPS&PS in 2022 and 2023.  

This project is run in Python 3.8, Pytorch-cuda=11.7. Be careful that to run these scripts, some input and output paths need to be adjusted or created.


    
Train Model
  
        cd Model
  1. Choose the model, e.g. resnet.
        vi train_<model_name>.sh
  2. set up your environment in the train_<model_name>.sh
        source <your conda env> 
        source train_<model_name>.sh
  
   
