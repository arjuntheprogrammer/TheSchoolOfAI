# TSAI_Session11

Project for The School of AI
---------------------------------------------------------------------------------------

## Tasks
1. Pick this code https://github.com/amitdoda1983/EVA-Session-6/blob/master/6A_M6_Amit_Doda.ipynb
2. Add CutOut to this
3. Use LR Finder to find the best starting learning rate
4. Use SDG with Momentum
5. Train for 100 Epochs. [Keep Verbose = 0 (no logs)] 
6. Print the top accuracy
7. Show Training and Test Accuracy curves
8. Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
9. Submit

--------------------------------------------------------------------------------------

**We are getting validation accuracy of:**
1. Basic Model [Model1]: 83.05%
2. Model2:  86.06%
3. Model2 for next 50 epochs: 86.67%
4. Model2 after adding cutout: 86.43%
5. Model2 with SGD Momentum: 86.52%

--------------------------------------------------------------------------------------

**Notebook Name: Assignment11.ipynb**

--------------------------------------------------------------------------------------

**Model File Names:**
1. model1.h5 - Basic model without training
2. model1_afterTraining.h5 - Basic model with training
3. model2.h5 - Model2 without training
4. model2_afterTraining.h5 - Model2 with training and with Image Normalization,  Batch Normalization, L2 Regularizer,  Dropout
5. model2_afterTraining_next50.h5 - Model2 with training for next 50 epochs
6. model2_withCutout_afterTraining.h5 - Model2 with training with Cutout
7. model2_withCutout_withSGDMomentum_afterTraining.h5 - Model2 with training with Cutout with SGD Momentum

--------------------------------------------------------------------------------------
