# TSAI Session16

Project for The School of AI
---------------------------------------------------------------------------------------

# Objectives

1. Collect 100 images of faces from online sources (you can use any existing database as well, but we need multiple faces)
2. Please make sure that there are not too many faces in the image
3. Classes are:  Front, Left, Right, Up, Down, UpLeft, UpRight, DownLeft, DownRight, Top, Back. Please make sure you have these kind of faces in your collection. Also please make sure that your LEFT is the LEFT of your screen. 
4. resize your images to 400x400
5. Rename your images as img_001 to img_100. 
6. Annotate these objects using VGG Annotator (Links to an external site.) (using a local copy)
7. Use K-means clustering to find out the top 4 anchor boxes
8. Upload to github
  * images in a zipped folder
  * your annotation file (json)
  * k-means code
9. Add a readme file and show:
  * few screenshots of your annotations
  * your 4 bounding box dimensions

Note: You will be using these images for your Face Recognition Session, so make sure your annotations are good.

--------------------------------------------------------------------------------------
## Afer K-Means, 4 bounding box dimesions we get are:
1. 0.84026957 0.39908429
2. 0.60873609 0.3349399 
3. 0.76679704 0.75434331
4. 0.35124573 0.23816278

--------------------------------------------------------------------------------------
### Few Screenshots for annotation:
![SS2](https://user-images.githubusercontent.com/15984084/69374959-3ed1fe80-0ccd-11ea-80af-8ac8314d6c94.PNG)

![SS3](https://user-images.githubusercontent.com/15984084/69374961-3ed1fe80-0ccd-11ea-90bb-c7b750de98b2.PNG)

![SS4](https://user-images.githubusercontent.com/15984084/69374962-3ed1fe80-0ccd-11ea-983b-55580c88211d.PNG)

![SS1](https://user-images.githubusercontent.com/15984084/69374963-3ed1fe80-0ccd-11ea-973e-dd48cbe41076.PNG)

--------------------------------------------------------------------------------------

### Notebook Names: 
1. **Assignment16.ipynb**

--------------------------------------------------------------------------------------