# The School Of AI Assignment

* Part 1: Train this: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html for any other environment (except CartPole-v0):
    1. Upload the screenshot of the program working on your computer (some atari game must be shown)
    2. Share the flow-chart diagram of 11+ functions and the main "for" loop describing the training process and models.

* Part 2: Go through this code: https://github.com/ikostrikov/pytorch-a3c :
    1. Share the flow chart diagram for every function for the code - there should be 17+ functions including the main.py considered as function).

---
**What is Q-Learning?**

Q-learning is an off policy reinforcement learning algorithm that seeks to find the best action to take given the current state. It’s considered off-policy because the q-learning function learns from actions that are outside the current policy, like taking random actions, and therefore a policy isn’t needed. More specifically, q-learning seeks to learn a policy that maximizes the total reward.


**Deep Q-Networks**

In deep Q-learning, we use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output. The comparison between Q-learning & deep Q-learning is wonderfully illustrated below:

![image](https://user-images.githubusercontent.com/15984084/77251887-f32d6280-6c76-11ea-83bf-91ea46356d7c.png)

**Steps involved in deep Q-learning networks (DQNs)**
1. All the past experience is stored by the user in memory
2. The next action is determined by the maximum output of the Q-network
3. The loss function here is mean squared error of the predicted Q-value and the target Q-value – Q*. This is basically a regression problem. However, we do not know the target or actual value here as we are dealing with a reinforcement learning problem. Going back to the Q-value update equation derived fromthe Bellman equation.

![image](https://user-images.githubusercontent.com/15984084/77251950-6fc04100-6c77-11ea-93ac-b71098970fe6.png)

---

**Actor Critic Model**
* AC model has two aptly named components: an actor and a critic.
* The former takes in the current environment state and determines the best action to take from there. It is essentially what would have seemed like the natural way to implement the DQN.
* The critic plays the “evaluation” role from the DQN by taking in the environment state and an action and returning a score that represents how apt the action is for the state.
* We want to determine what change in parameters (in the actor model) would result in the largest increase in the Q value (predicted by the critic model).

![image](https://user-images.githubusercontent.com/15984084/77252427-cb8bc980-6c79-11ea-813c-130a07ef51f8.png)

---

**Asynchronous Advantage Actor-Critic (A3C)**
* A3C’s released by DeepMind in 2016 and make a splash in the scientific community.
* It’s simplicity, robustness, speed and the achievement of higher scores in standard RL tasks made policy gradients and DQN obsolete.
* The key difference from A2C is the Asynchronous part. A3C consists of multiple independent agents(networks) with their own weights, who interact with a different copy of the environment in parallel.
* Thus, they can explore a bigger part of the state-action space in much less time.

![image](https://user-images.githubusercontent.com/15984084/77252293-dabe4780-6c78-11ea-8c05-f3cdb4c99a7e.png)

* The agents (or workers) are trained in parallel and update periodically a global network, which holds shared parameters.
* The updates are not happening simultaneously and that’s where the asynchronous comes from.
* After each update, the agents resets their parameters to those of the global network and continue their independent exploration and training for n steps until they update themselves again.





---

## Screenshots

![Phase2Session8](https://user-images.githubusercontent.com/15984084/77251593-3686d180-6c75-11ea-8009-0989a7fb575d.png)


![Reinforcement Learning (DQN)](https://user-images.githubusercontent.com/15984084/77251590-31298700-6c75-11ea-97be-8dc9c3a4c9d2.jpg)

![AC3](https://user-images.githubusercontent.com/15984084/77251606-47cfde00-6c75-11ea-88ca-1a7d00c23e17.png)

---