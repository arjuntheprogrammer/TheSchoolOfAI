# TSAI_Session8-9

Project for The School of AI
---------------------------------------------------------------------------------------
## **Part 1 :**

1. Take your 6A Code (your optimized version, not the base version with Dense layer), and learn how to integrate **gradCAM** with your code. 
> Learn more about gradCAM here - https://www.hackevolve.com/where-cnn-is-looking-grad-cam/
2. As proof of integration, test 4 images (remember the image ids or names) from your network and show the visualization like this:
![image](https://user-images.githubusercontent.com/15984084/64717395-e074ec80-d4e1-11e9-8867-3df9836f5b12.png)

3. This is the first part of the assignment. 
---------------------------------------------------------------------------------------

## **Part 2 :**

1. Train your 6A model again, but this time add **CutOut**. 
> Use this link for reference - https://github.com/yu4u/cutout-random-erasing
2. Show the same 4 images again with gradCAM's result. 
3. This is the second part of the assignment

--------------------------------------------------------------------------------------

### We are getting validation **accuracy of  82.93%** after using Cutout.

--------------------------------------------------------------------------------------

### Notebook Names: 
1. **Assignment8-9.ipynb**

--------------------------------------------------------------------------------------

### Model File Names:
1. model.h5 - Basic model without training
2. model_afterTraining.h5 - Basic model with training
3. model_withCutout_afterTraining.h5 - Trained Model on cutout images

--------------------------------------------------------------------------------------
