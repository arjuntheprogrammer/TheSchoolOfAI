# Self Driving Car

# Importing the libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
import cv2

# Importing the TD3 object from our AI in ai.py
from ai import TD3, ReplayBuffer

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

##################################################
''' We set the parameters '''
seed = 0 # Random seed number
start_timesteps = 1000
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
action2rotation = [0,5,10,-5,-10]
last_reward = 0

##################################################
''' We initialize the variables '''
total_timesteps = 0
episode_num = 0
done = False
t0 = time.time()
episode_timesteps = 0
##################################################
''' We set seeds and we get the necessary information
on the states and actions in the chosen environment'''
state_dim = 5
action_dim = 5
action_space_low = -5
action_space_high = 5
max_action = 1

# Initializing the last distance
last_distance = 0
orientation = 0

obs_img = np.zeros((50, 50))
obs_dis = last_distance
obs_ori = orientation

new_obs_img = np.zeros((50, 50))
new_obs_dis = last_distance
new_obs_ori = orientation

##################################################
''' We create the policy network (the Actor model) '''
policy = TD3(state_dim, action_dim, max_action)

##################################################
''' We create the Experience Replay memory'''
replay_buffer = ReplayBuffer()

##################################################
im = CoreImage("./images/MASK1.png")

imgCV2 = cv2.imread('./images/MASK1.png')
rows,cols, dims = imgCV2.shape
# print("rows = ", rows, " cols = ", cols, " dims=", dims)

# Initializing the map
first_update = True

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global swap
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1420
    goal_y = 622
    first_update = False
    swap = 0


# Creating the car class
class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation


# Creating the game class
class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def step(self, rotation):
        global last_reward
        global last_distance
        global goal_x
        global goal_y
        global swap

        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        if sand[int(self.car.x),int(self.car.y)] > 0: # car on sand
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            last_reward = -5
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = 5
            if distance < last_distance:
                last_reward = 7

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -10

        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -10

        if self.car.y < 5:
            self.car.y = 5
            last_reward = -10

        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -10

        if distance < 25:
            done=True
            print("########GOAL REACHED########")
            last_reward = 10
            if swap == 1:
                print("########GOAL 1 REACHED########")
                goal_x = 1420
                goal_y = 622
                swap = 0
            else:
                print("########GOAL 2 REACHED########")
                goal_x = 9
                goal_y = 85
                swap = 1
        else:
            done=False
        last_distance = distance


    def update(self, dt):

        global policy
        global last_reward
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global done
        global episode_timesteps
        global total_timesteps

        global obs_img
        global obs_ori
        global obs_dis
        global new_obs_img
        global new_obs_ori
        global new_obs_dis

        global imgCV2

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        cv2_y = int(self.height) - (int(self.car.y+50)) , int(self.height) - (int(self.car.y)-50)
        cv2_x = int(self.car.x-50), int(self.car.x)+50

        img = np.mean(imgCV2, axis=2)
        imgCV2_temp = np.zeros((100, 100), dtype=float)
        indX = -1
        indY = -1
        for i in range(cv2_y[0], cv2_y[1]):
            indY+=1
            indX = -1
            for j in range(cv2_x[0], cv2_x[1]):
                indX+=1
                if(i<0 or j<0 or i >= self.height or j >=self.width):
                    imgCV2_temp[indY][indX] = 255
                else:
                    imgCV2_temp[indY][indX] = img[i][j]
        imgCV2_temp = cv2.resize(imgCV2_temp, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("imgCV2_temp", imgCV2_temp)
        new_obs_img = imgCV2_temp

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        new_obs_ori = Vector(*self.car.velocity).angle((xx,yy))/180.0
        new_obs_dis = last_distance

        action = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < start_timesteps:
            action = random.sample(action2rotation, 1)[0]

        else: # After 10000 timesteps, we switch to the model
            if(total_timesteps%1000==0):
                for loop in range(10):
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                episode_timesteps = 0
            action = policy.select_action(new_obs_img, new_obs_ori, -new_obs_ori, new_obs_dis)
            action = action[0].item()

        self.step(action)

        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((obs_img, obs_ori, -obs_ori, obs_dis, new_obs_img, new_obs_ori, -new_obs_ori, new_obs_dis, action, last_reward, done))

        obs_img = new_obs_img
        obs_ori = new_obs_ori
        obs_dis = new_obs_dis

        total_timesteps += 1
        episode_timesteps += 1

        if(last_reward>-5):
            print("")
            print("**************")
            print("car location = ", self.car.x, " ", self.car.y)
            print("action = ", action)
            print("last_reward = ", last_reward)
            print("last_distance = ", last_distance)
            print("total_timesteps = ", total_timesteps)


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
