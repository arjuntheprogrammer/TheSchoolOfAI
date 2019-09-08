# TSAI_Session6

Project  for The School of AI
---------------------------------------------------------------------------------------
## **Assignment6A :**

1. Remove Dense Layer
2. Add layers required to reach RF
3. Fix kernel scaleup and down (1x1)
4, See if all dropouts are properly placed
5. Follow the guidelines we discussed in the class ( Get accuracy more than the base accuracy in less number 100 epochs. Hint, you might want to use "padding='same',".
6. Save File as Assignment 6A.

---------------------------------------------------------------------------------------
## **Assignment6B :**

1. Normal Convolution
2. Spatially Separable Convolution (Conv2d(x, (3,1)) followed by Conv2D(x,(3,1))
3. Depthwise Separable Convolution
4. Grouped Convolution (use 3x3, 5x5 only)
5. Grouped Convolution (use 3x3 only, one with dilation = 1, and another with dilation = 2)
6. You must use all of the 5 above at least once
7. Train this new model for 50 epochs.
8. Save File as Assignment 6B
--------------------------------------------------------------------------------------
## Notebook Names: 
1. **Assignment6A.ipynb**
2. **Assignment6B.ipynb**
--------------------------------------------------------------------------------------
### Model File Names:
#### For 6A:
1. model.h5 and model_afterTraining.h5
2. model1.h5 and model_afterTraining1.h5
3. model2.h5 and model_afterTraining2.h5
4. model3.h5 and model_afterTraining3.h5
5. model4.h5 and model_afterTraining4.h5
6. model5.h5 and model_afterTraining5.h5
7. model6.h5 and model_afterTraining6.h5
8. model7.h5 and model_afterTraining7.h5 

#### For 6B:
1. model.h5 and model_afterTraining.h5
2. model_Spatially_Separable1.h5 and model_Spatially_Separable1_afterTraining.h5
3. model_Depth_Separable1.h5 and model_Depth_Separable1_afterTraining.h5
4. model_GroupedConvolution1.h5 and model_GroupedConvolution1_afterTraining.h5
5. model_GroupedConvolution2.h5 and model_GroupedConvolution2_afterTraining.h5

---------------------------------------------------------------------------------------
### Group Convolution Model 1 Architecture
![model_GroupedConvolution1_plot](https://user-images.githubusercontent.com/15984084/64485709-57d82100-d241-11e9-8a79-60432fe473d6.png)
---------------------------------------------------------------------------------------
### Group Convolution Model 2 Architecture
![model_GroupedConvolution2_plot](https://user-images.githubusercontent.com/15984084/64485710-57d82100-d241-11e9-9fff-d6890d12e704.png)

---------------------------------------------------------------------------------------
