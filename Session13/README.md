# TSAI Session13

Project for The School of AI
---------------------------------------------------------------------------------------

# Objectives

1. Design the ResNet18 model.
    * Your model must look like Conv->B1->B2->B3->B4 and not individually called Convs.
2. If not already using, then:
    * Use Batch Size 128
    * Use Normalization values of: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    * Random Crop of 32 with padding of 4px
    * Horizontal Flip (0.5)
    * Optimizer: SGD, Weight-Decay: 5e-4
    * OneCycleLR
    * use cutout if possible
3. Save model (to drive) after every 50 epochs or best model till now
4. Describe your blocks, and the stride strategy you have picked
5. Train for 300 Epochs
6. Assignment Target Accuracy is 90%, so exit gracefully if you reach 90% (you can target more, it can go till ~93%)

Note:
* you can use Fenwicks library
* use all the knowlegde gained from the Assignment 12

--------------------------------------------------------------------------------------


## The best validation accuracy achieved = 91.15% in 95th Epoch
* val_loss: 0.5345
* val_acc: 0.9115
---

**Strategy Used:**
  * we used 4 convolution blocks, each conv. block started with batchNorm and Relu activation, followed by a 2 convolution layer(each with stride of 2) in parallel to the projection shortcut which got added together.
  * then again starting with batchNorm and Relu activation, followed by a 2 convolution layer(each with stride of 2) in parallel to the identity shortcut this time which got added together.
--------------------------------------------------------------------------------------

### Notebook Names: 
1. **Assignment13.ipynb**

--------------------------------------------------------------------------------------

### Model File Names:
1. cifar10_Resnet18_model.095.h5 - Model with the best validation accuracy.


--------------------------------------------------------------------------------------
