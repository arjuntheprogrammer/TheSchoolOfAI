# TSAI_Session10

Project for The School of AI
---------------------------------------------------------------------------------------

1. Refer to the GRADCAM code we wrote: https://colab.research.google.com/drive/10GugXUNI7ztK2joRZUnYyqRrQbYnOQE0
2. Build GradCAM images for the one layer before the one we used, and one layer before this one. Show the results.
3.Load this image: https://user-images.githubusercontent.com/15984084/65062738-54a80800-d99a-11e9-9fb9-92a5dc7724a7.jpg
4. "Find" "sunglasses" in the image using GradCAM


--------------------------------------------------------------------------------------

### Layers used for GRADCAM Visualization:
1. block5_conv1
2. block5_conv2

--------------------------------------------------------------------------------------

### Steps involved:

1. Load a pre-trained model
2. Load an image which can be processed by this model (224x224 for VGG16 why?)
3. Infer the image and get the topmost class index
4. Take the output of the final convolutional layer
5. Compute the gradient of the class output value w.r.t to L feature maps
6. Pool the gradients over all the axes leaving out the channel dimension
7. Weigh the output feature map with the computed gradients (+ve)
8. Average the weighted feature maps along channels
9. Normalize the heat map to make the values between 0 and 1

--------------------------------------------------------------------------------------

### Notebook Names: 
1. **Assignment10.ipynb**

--------------------------------------------------------------------------------------

## We are successfully able to highlight the area using GRADCAM where the Sunglasses were present in the given picture

--------------------------------------------------------------------------------------
