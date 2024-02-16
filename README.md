# CEPC-AHCAL-PID
Particle Identification Using Artificial Neural Network. This net is trained on Geant 4 simulation results with only three kinds of incident particles: muon+, e+, pion+. Their energy is consistent with the particles'energy collected in 2022 CERN beam test.  

This project is run in Python 3.8, Pytorch-cuda=11.7. Be careful that in order to run these scripts, some input and output path need to be adjusted or created.

Primary steps are provided below. Datasets could be made in step 1, and once datasets are available, step 1, which is mainly for making own datasets, could be bypassed. If using evaluate function in Evaluate.py, extra test sets must be prepared. 

When running Train.py, the default hyper-parameters would be transported into it from Config.config.py, and of course deciding and testing various hyper-parameters are recommened. After Train.py finishes running, the net and some evaluation results would be saved in the Checkpoint directory. 



1. Prepare Datasets

    1). Convert *.root file to *.npy file: 

        cd MakeDataSets
        python MakeTrainSets.py
  
    2). Make train, validation, test set( default 8:1:1)
    
        cd MakeDataSets
        python MakeFinalDataSet.py
    
2. Train Model
  
        cd Model
        python Train.py
  
   
