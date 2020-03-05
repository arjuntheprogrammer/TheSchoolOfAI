# The School Of AI Assignment

1. Create a new map of some other city for the code shared above
2. Add a DNN with 1 more FC layer.
3. Your map must have 3 target A1>A2>A3 and your car/robot/object must target these alternatively.
4. Train your best model and upload a video on Youtube and share URL on P2S7
5. Answer these questions in S7-Assignment-Solution:
    1. What happens when "boundary-signal" is weak when compared to the last reward?
    2. What happens when Temperature is reduced?
    3. What is the effect of reducing LaTeX: \gammaγ?
6. Heavy marks for creativity, map quality, targets and other things.
** If you use the same maps or have just replicated shared code, you will get 0 for this assignment and -50% advance deduction for next assignment.

---


## DEEP Q LEARNING

    In deep Q-learning, we use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output.

**Experience Replay**

    Instead of performing a network update immediately after each “experience” (action, state, reward, following state), these experiences are stored in memory and sampled from randomly.

## Steps
![image](https://user-images.githubusercontent.com/15984084/76006079-f5cb5080-5f31-11ea-9480-26814cb6fd35.png)
1. All the past experience is stored by the user in memory.
2. The next action is determined by the maximum output of the Q-network.
3. The loss function here is mean squared error of the predicted Q-value and the target Q-value – Q*. This is basically a regression problem. However, we do not know the target or actual value here as we are dealing with a reinforcement learning problem. Going back to the Q-value update equation derived fromthe Bellman equation.
![image](https://user-images.githubusercontent.com/15984084/76006162-18f60000-5f32-11ea-873b-2f9eebfae204.png)


---

## Installation

**For Linux and MacOS**

* $ conda create -n aiml_py27
* $ conda activate aiml_py27
* $ conda install pytorch==0.3.1 -c pytorch
* $ conda install matplotlib
* $ conda install -c conda-forge kivy
* $ pip install pygame

**For Windows**
* $ conda install -c peterjc123 pytorch-cpu
* $ conda install -c conda-forge kivy

---

## Application Description
Trained Hospital Van to take the Corona Virus-Infected Patients to the Hospital in a simulated environment.
- Kivy used for the UI
- Python used for training the neural network
- Chandigarh City Map is used for mapping the roads to the map.
- There are 3 sensor put on the van, which identifies the pixel density in front of it - which tell if there is a road it front of the van. 1st sensor to see the road ahead, other two to detect if there is any turn present on the road.
- Van first picks up the Patient1, then Patient 2, and the goes to the Hospital and this continues infinitely.
- But during this course, actions taken by the Hospital Van gets continously improved.
    - After a short time Van start taking the shortest of distances to reach the goal.
    - It stays more on the road, less on the part where there are no roads.
    - Stays away from boudary of the map so that it never crosses it.


---

## Youtube Link
> https://youtu.be/T5cy_siR1xw

![Deep Q Learning - Simulation](https://user-images.githubusercontent.com/15984084/76007007-5dce6680-5f33-11ea-9f50-8138f07447e8.png)

---