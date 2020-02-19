# TSAI Session 24

Project for The School of AI
---------------------------------------------------------------------------------------

# Objectives

1. Continuing our objective with learning Pytorch, refer to this code: LINK https://colab.research.google.com/drive/13tb_Mb4oaJkFtdadqD2pHx_bqdn46Vsp

2. Change this model in such a way that it:
    * less than 15k params
    * uses dropout of 0.1 
    * uses batchnorm
    * uses randomrotate transform
    * uses StepLR with step size = 6 and gamma = 0.1
    * achieves 99.3% test accuracy
    * less than 15 epochs. 
3. Once you're ready, fill in P2S5 Quiz. Also, upload to your PUBLIC GitHub account and be ready with the link. 

--------------------------------------------------------------------------------------

# Learnings
1. Optimise PyTorch Model
2. Apply transforms using PyTorch's inbuild functions
3. StepLR Implementation
4. Building efficient model with less parameters and high accuracy.

--------------------------------------------------------------------------------------

### Notebook Names: 
**EVAP2S5_PyTorch.ipynb**

---

## Final Results
## In Epoch 12 we were able to achieve 99.32% accuracy.

* Total params: 14,980
* used dropout of 0.1
* used batchnorm
* uses randomrotate transform with value of 18
* uses StepLR with step size = 6 and gamma = 0.1

---