# TSAI Session16-Part2

Project for The School of AI
---------------------------------------------------------------------------------------

# Objectives

1. Refer to this TSAI COLAB FILE (https://colab.research.google.com/drive/1iZdzI0VEG8ieRgHXKT7tNEE7iN3gw4tN)
2. We have stitched multiple projects into 1, so you can train tiny-YOLO on COLAB!

3. Refer to this blog: LINK (https://medium.com/@today.rafi/train-your-own-tiny-yolo-v3-on-google-colaboratory-with-the-custom-dataset-2e35db02bf8f) and LINK (https://github.com/rafiuddinkhan/Yolo-Training-GoogleColab/blob/master/helmet.ipynb). This is the main source for our project. 

4. Refer to the "main.py" in Step 1. 

5. Here is what you need to do:
    * create your own dataset using main.py
    * collect 200 images (more the better) for any 1 class of your choice. e.g. this project is for the helmet. You cannot use the same dataset as in this project. 
    * you should be able to find a short youtube video for this class as well (example classes you can pick: traffic_light, dog, car, bird, flag, etc)
    * annotate 200 images as explained in the links above
    * replace the data_for_colab folder with your custom folder
    * train YOLO for 1000 epochs (more the better)
  download 1 youtube video  which has your class, you can use this: https://www.y2mate.com/en4

6. run this command
> !./darknet detector demo data_for_colab/obj.data data_for_colab/yolov3-tiny-obj.cfg backup/yolov3-tiny-obj_1000.weights  -dont_show youtube_video.mp4 -i 0 -out_filename veout.avi

7. upload your video on YouTube.
8. Share the link with us (and your LinkedIn as well if you want!)!
9. Fixed deadline and try to do this in a group (at least share your datasets). 

---

### Notebook Names: 
**EVA2_YOLO.ipynb**

---

## YouTube Link for Output Videos:
1. https://youtu.be/rUC6eHQSlOw
2. https://youtu.be/wlduip6uWa0

---