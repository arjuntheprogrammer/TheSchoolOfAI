## Facial landmark detector with dlib

* While the 68-point detector localizes regions along the eyes, eyebrows, nose, mouth, and jawline, the 5-point facial landmark detector reduces this information to:
  * 2 points for the left eye
  * 2 points for the right eye
  * 1 point for the nose

* Detecting facial landmarks in an image is a two step process:
  1. First we must localize a face(s) in an image. This can be accomplished using a number of different techniques, but normally involve either Haar cascades or HOG + Linear SVM detectors (but any approach that produces a bounding box around the face will suffice).
  2. Apply the shape predictor, specifically a facial landmark detector, to obtain the (x, y)-coordinates of the face regions in the face ROI.

* Given these facial landmarks we can apply a number of computer vision techniques, including:
  * Face part extraction (i.e., nose, eyes, mouth, jawline, etc.)
  * Facial alignment
  * Head pose estimation
  * Face swapping
  * Blink detection

## Output Images
![img1](https://user-images.githubusercontent.com/15984084/69754239-6f0d1780-117b-11ea-86bd-1698bfd5249c.png)
![img2](https://user-images.githubusercontent.com/15984084/69754240-6f0d1780-117b-11ea-85bb-857428a57a5f.png)
![img3](https://user-images.githubusercontent.com/15984084/69754241-6fa5ae00-117b-11ea-92c4-4c1cb4b88287.png)

Reference: 
* https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/
* https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
* https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
