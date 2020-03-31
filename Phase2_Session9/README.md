# The School Of AI Assignment

---

## **Phase 2 Session 9: Assignment**

1. Well, there is a reason why this code is in the image, and not pasted.
2. You need to:
    1. write this code down on a Colab file, upload it to GitHub.
    2. write a Readme file explaining all the 15 steps we have taken:
        1. read me must explain each part of the code
        2. each part of the code must be accompanied with a drawing/image (you cannot use the images from the course content)
    3. Upload the link.

---
## **Twin Delayed DDPG (TD3)**
* **DDPG** stands for **Deep Deterministic Policy Gradient** and is a recent breakthrough in AI, particularly in the case of environments with continuous action spaces.
* To be able to apply Q-learning to continuous tasks, the authors introduced the Actor-Critic model.
* Actor-Critic has 2 neural networks that the following way:
    1. The Actor is the policy that takes as input the State and outputs Actions
    2. The Critic takes as input States and Actions concatenated together and outputs a Q-value
* The Critic learns the optimal Q-values which are then used to for gradient ascent to update the parameters of the Actor.
* By combining learning the Q-values (which are rewards) and the parameters of the policy at the same time, we can maximize expected reward.


---

## **TD3 Steps With Screenshots**

<img width="1083" alt="Screenshot 2020-03-31 at 1 35 11 PM" src="https://user-images.githubusercontent.com/15984084/78004576-ec2de080-7357-11ea-941c-6a9987ba3073.png">

### **Step1**: Define Experience Replay Memory
<img width="922" alt="Screenshot 2020-03-31 at 1 17 51 PM" src="https://user-images.githubusercontent.com/15984084/78001064-cd791b00-7352-11ea-960f-10ba8eb25a0b.png">

### **Step2**: Define Actor Model
<img width="717" alt="Screenshot 2020-03-31 at 1 17 59 PM" src="https://user-images.githubusercontent.com/15984084/78001083-d2d66580-7352-11ea-9fc1-524c72bb431a.png">

### **Step3**: Define Critic Model
<img width="817" alt="Screenshot 2020-03-31 at 1 19 10 PM" src="https://user-images.githubusercontent.com/15984084/78001100-d964dd00-7352-11ea-8bd4-43c918b77d83.png">

### **Step4**: Get Random sample from Experience Replay Memory
<img width="1054" alt="Screenshot 2020-03-31 at 1 19 30 PM" src="https://user-images.githubusercontent.com/15984084/78001122-e255ae80-7352-11ea-98cd-b151529fafd7.png">

<img width="973" alt="Screenshot 2020-03-31 at 1 19 40 PM" src="https://user-images.githubusercontent.com/15984084/78001146-eb468000-7352-11ea-81ee-c1542a3dbd43.png">

### **Step5**: Get Next Action (a’) from Next State(s’) using actor target
<img width="895" alt="Screenshot 2020-03-31 at 1 19 48 PM" src="https://user-images.githubusercontent.com/15984084/78001160-ef729d80-7352-11ea-8e07-7a3e892a5abe.png">

### **Step6**: Gaussian Noise to Next Action (a’) and clamp to a range
<img width="1039" alt="Screenshot 2020-03-31 at 1 20 04 PM" src="https://user-images.githubusercontent.com/15984084/78001188-f7cad880-7352-11ea-8fc5-413ae635421a.png">

### **Step7**: Two Critic Target take s’ and a’ as input and return two Q Values
<img width="1012" alt="Screenshot 2020-03-31 at 1 20 14 PM" src="https://user-images.githubusercontent.com/15984084/78001226-03b69a80-7353-11ea-9c54-540a04c65a87.png">

### **Step8**: Take Minimum of Q1 and Q2 and output Target Q
<img width="784" alt="Screenshot 2020-03-31 at 1 20 22 PM" src="https://user-images.githubusercontent.com/15984084/78001254-0fa25c80-7353-11ea-9e40-06181bf62999.png">


### **Step9**: Get the final target of the two Critic models
<img width="979" alt="Screenshot 2020-03-31 at 1 20 28 PM" src="https://user-images.githubusercontent.com/15984084/78001284-1d57e200-7353-11ea-85a1-7659369569cf.png">


### **Step10**: Two Critic Model take current state(s) and current action (a) as input and return two current Q Values
<img width="1007" alt="Screenshot 2020-03-31 at 1 20 35 PM" src="https://user-images.githubusercontent.com/15984084/78001301-22b52c80-7353-11ea-8db9-0b916c151367.png">

### **Step11**: Compute the Critic Loss
<img width="831" alt="Screenshot 2020-03-31 at 1 20 40 PM" src="https://user-images.githubusercontent.com/15984084/78001415-44aeaf00-7353-11ea-9dab-99b4f0056497.png">

### **Step12**: Backpropagate the critic loss and update the parameters of two Critic models
<img width="1133" alt="Screenshot 2020-03-31 at 1 20 49 PM" src="https://user-images.githubusercontent.com/15984084/78001456-51cb9e00-7353-11ea-9288-3223cba43a1f.png">

### **Step13**:  Once every two iterations, we update our Actor model
<img width="1242" alt="Screenshot 2020-03-31 at 1 21 19 PM" src="https://user-images.githubusercontent.com/15984084/78001508-6740c800-7353-11ea-844b-1be0fb7d75d4.png">

### **Step14**: Still, in once every two iterations, update our Actor Target by Polyak Averaging
<img width="1247" alt="Screenshot 2020-03-31 at 1 21 28 PM" src="https://user-images.githubusercontent.com/15984084/78001623-97886680-7353-11ea-977e-8cb9a77f1eda.png">

### **Step15**: Still, in once every two iterations, we update our Critic  Target by Polyak Averaging
<img width="1226" alt="Screenshot 2020-03-31 at 1 21 43 PM" src="https://user-images.githubusercontent.com/15984084/78001797-d9b1a800-7353-11ea-91a0-e7d6e9169dbb.png">

---