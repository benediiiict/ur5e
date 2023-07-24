#!/usr/bin/env python3

import numpy as np
import gym
import os
import rospy
import RL_env
import math
# import actionlib
# import torch
# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common import make_vec_env

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from numpy import random
from geometry_msgs.msg import Point, Pose
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from std_msgs.msg import String
from math import pi
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import Vector3, Wrench, WrenchStamped
from gym import spaces
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from env_client import UrMoveClient

class PIHAssembleEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the agent must learn to go always left. 
  """
  metadata = {'render.modes': ['human']}
  # Define constants for clearer code

  def __init__(self):
    super(PIHAssembleEnv, self).__init__()

    self.robot = UrMoveClient()
    self.action_dim = 3
    self.obs_dim = 9
    self.count = 0
    self.target = np.array([0.5490, 0.1595, 0.2000]) 
    self.position_ready = np.array([0.5532, 0.1595, 0.2500]) 
    # self.target = np.array([-0.13357, 0.6015, 0.2000])            ###################### Real ######################
    # self.position_ready = np.array([-0.13367, 0.6000, 0.2700])    # Real       跟pose_goal差   ( y -0.002, z -0.005 ) 
    self.pose_last = []
    self.distance = []
    self.ft_sensor_array = []
    self.force = []
    self.force_again = []
    self.hole = []
    self.max_force = 0.0
    self.max_torque = 0.0

    high = np.ones([self.action_dim])
    self.action_space = spaces.Box(np.float32(-high), np.float32(high))
    # self.action_space = spaces.Box(-high, high, dtype=np.float32)
    high = np.inf*np.ones([self.obs_dim])
    self.observation_space = spaces.Box(np.float32(-high), np.float32(high))
    # self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    rospy.Subscriber('/ft_sensor_topic', WrenchStamped, self.force_return)    # force_feedback
    rospy.Subscriber('goalpoint', Point, self.point_return)

########################################### User-Defined Function ################################################

  def action_normalization(self, action):     ############### Real ###############
    action[0] = action[0] * 0.002                         # 0.003 + 0.0008
    action[1] = action[1] * 0.002                         # 0.0001
    action[2] = action[2] * 0.003                         # 0.001
    return action
  
  def force_feedback(self,data):
        # rospy.loginfo(data)
        self.adjust = np.array([data.wrench.force.x,\
                                data.wrench.force.y,\
                                data.wrench.force.z,\
                                data.wrench.torque.x,\
                                data.wrench.torque.y,\
                                data.wrench.torque.z])
        # self.adjust[2] -= 11.898
        self.ft_sensor_array = self.adjust
        
  def point_return(self, data):
    self.point = []
    self.point.append(data.x)
    self.point.append(data.y)
    self.hole = self.point

  def force_return(self, data):
    self.force_data = []
    self.force_data.append(data.wrench.force.x)
    self.force_data.append(data.wrench.force.y)
    self.force_data.append(data.wrench.force.z)
    self.ft_sensor_array = self.force_data
    # print(self.force_data)
    # return self.ft_sensor_array
  
  def Dis_plus_ForceFeedback(self, dis, pose_now, ft_x, ft_y, ft_z):
    self.obs = np.array([dis[0], dis[1], dis[2], pose_now.position.x, pose_now.position.y, pose_now.position.z, ft_x, ft_y, ft_z])      # self.ft_sensor_array[0], self.ft_sensor_array[1], self.ft_sensor_array[2]
    return self.obs
  
  def cal_reward_dis(self, dis):
    distance_r = math.sqrt(pow(dis.position.x - self.target[0], 2) + pow(dis.position.y - self.target[1], 2) + pow(dis.position.z - self.target[2], 2))
    return distance_r
  
  def Gauss(self):       # x,y,z 跟 pose_goal 的值一樣
    x = random.normal(loc = -0.13357, scale=0.0002, size=(1, 1))
    # xf = x.astype(float)
    y = random.normal(loc = 0.6015, scale=0.0002, size=(1, 1))
    # yf = y.astype(float)
    z = random.normal(loc = 0.2750, scale=0.0001, size=(1, 1))
    # zf = z.astype(float)
    # pose_Gauss = np.array([xf, yf, zf], dtype='float')
    pose_Gauss = np.array([np.float32(x), np.float32(y), np.float32(z)]).astype(np.float32)
    # pose_Gauss = [x, y, z]
    return pose_Gauss

########################################### Stablebaseline3 ##################################################

  def step(self, action):
    # err_msg = f"{action!r} ({type(action)}) invalid"
    # assert self.action_space.contains(action), err_msg
    print("=============================one step================================")
    done = False
    reward = -1
    force_flag = 0
    self.count += 1
    self.action = self.action_normalization(action)
    print(self.action)
    print("============================= action ================================ ")

    self.robot.move_arm(self.action[0], self.action[1], self.action[2])
    self.pose = self.robot.catch_current_pose()
    self.force = self.ft_sensor_array

    # self.reward_distance = self.cal_reward_dis(self.pose)
    # self.reward_distance_last = self.cal_reward_dis(self.pose_last)

    # self.force_again = self.ft_sensor_array
    # print("--------------------------- pose_last ======================== \n", self.pose_last)
    # print("--------------------------- pose_now  ======================= \n", self.pose)
    # print("--------------------------- force_now ========================= \n", self.force)
    # print("--------------------------- force_again_now ========================= \n", self.force_again)

    # random point             target = [0.5490, 0.1595, 0.2000]          real target = [-0.13357, 0.6015, 0.2000 (2700)]   
    if self.pose.position.x > (self.target[0] - 0.0012) and self.pose.position.x < (self.target[0] + 0.0012) and self.pose.position.y > (self.target[1] - 0.0012) and self.pose.position.y < (self.target[1] + 0.0012) and self.pose.position.z > (self.target[2] + 0.013) and self.pose.position.z < (self.target[2] + 0.018):
      reward = 5
      done = True
      print("doooooooooooooooooooooooooooooooooooooooone!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
      if self.force[0] > 20 or self.force[0] < -20 :
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! break !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        reward -= 3
        done = True
      elif self.force[1] > 20 or self.force[1] < -20 :
        reward -= 3
        done = True
      elif self.force[2] > 20 or self.force[2] < -15 :
        reward -= 3
        done = True
      else:
        if abs(self.pose.position.z - self.target[2]) < abs(self.pose_last.position.z - self.target[2]) :
          reward += abs(self.pose.position.z - self.pose_last.position.z) * 500
        elif abs(self.pose.position.z - self.target[2]) > abs(self.pose_last.position.z - self.target[2]) :
          reward -= abs(self.pose.position.z - self.pose_last.position.z) * 2000
          done = True
        elif self.pose.position.z > (self.target[2] + 0.015) and self.pose.position.z < (self.target[2] + 0.024):
          reward += 1.5
        elif self.pose.position.z < (self.target[2] + 0.014) or self.pose.position.z > (self.target[2] + 0.072):
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Wrong !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          reward -= 3
          done = True
        
        # if abs(self.pose.position.x - self.target[0]) < abs(self.pose_last.position.x - self.target[0]) :
        #   reward += abs(self.pose.position.x - self.pose_last.position.x) * 300
        # elif abs(self.pose.position.x - self.target[0]) > abs(self.pose_last.position.x - self.target[0]) :
        #   reward -= abs(self.pose.position.x - self.pose_last.position.x) * 300
        # elif self.pose.position.x > (self.target[0] - 0.0035) and self.pose.position.x < (self.target[0] + 0.0025):
        #   reward += 1  
        # elif self.pose.position.x < (self.target[0] - 0.010) or self.pose.position.x > (self.target[0] + 0.012):
        #   reward -= 3
        #   done = True

        ################### Gazebo #####################

        # if (self.target[0] + 0.0012) > self.pose.position.x > (self.target[0] - 0.0018):
        #   reward += 1.0
        # elif (self.target[0] + 0.0050) > self.pose.position.x > (self.target[0] + 0.0012):
        #   reward += 0.5
        # elif (self.target[0] + 0.0100) > self.pose.position.x > (self.target[0] + 0.0050):
        #   reward += 0.2
        # elif (self.target[0] - 0.0018) > self.pose.position.x > (self.target[0] - 0.0045):
        #   reward += 0.5  
        # elif (self.target[0] - 0.0045) > self.pose.position.x > (self.target[0] - 0.0080):
        #   reward += 0.2
        # elif self.pose.position.x < (self.target[0] - 0.008) or self.pose.position.x > (self.target[0] + 0.010):
        #   reward -= 1
        # if self.pose.position.x < (self.target[0] - 0.010) or self.pose.position.x > (self.target[0] + 0.012):
        #   reward -= 3
        #   done = True

        # # if (self.target[1]) < self.pose.position.y < (self.target[1] + 0.0005):
        # #   reward += 0.5
        # #   if self.pose.position.y < (self.target[1] - 0.0005):
        # #     reward -= 2
        # # elif (self.target[1] + 0.0005) < self.pose.position.y < (self.target[1] + 0.001):
        # #   reward += 0.3
        # # elif (self.target[1] + 0.0010) < self.pose.position.y < (self.target[1] + 0.0015):
        # #   reward += 0.1
        # # elif self.pose.position.y > (self.target[1] + 0.0015):
        # #   reward -= 2
        # if self.pose.position.y < (self.target[1] - 0.0006) or self.pose.position.y > (self.target[1] + 0.0006):
        #   reward -= 1

        # if self.pose.position.z < (self.target[2] + 0.025):
        #   reward += 1.0
        #   if self.pose.position.z < (self.target[2] + 0.015):
        #     reward -= 2
        #     done = True
        # elif (self.target[2] + 0.025) < self.pose.position.z < (self.target[2] + 0.045):
        #   reward += 0.8
        # elif (self.target[2] + 0.045) < self.pose.position.z < (self.target[2] + 0.065):
        #   reward += 0.6
        # elif (self.target[2] + 0.065) < self.pose.position.z < (self.target[2] + 0.085):
        #   reward += 0.4
        # elif (self.target[2] + 0.085) < self.pose.position.z < (self.target[2] + 0.105):
        #   reward += 0.2
        # elif self.pose.position.z > (self.target[2] + 0.125):
        #   reward -= 2

        ################## Real Env #####################

        if (self.target[0] + 0.0012) > self.pose.position.x > (self.target[0] - 0.0012):
          reward += 0.5
        elif (self.target[0] + 0.0040) > self.pose.position.x > (self.target[0] + 0.0012):
          reward += 0.25
        elif (self.target[0] + 0.0065) > self.pose.position.x > (self.target[0] + 0.0040):
          reward += 0.1
        elif (self.target[0] - 0.0012) > self.pose.position.x > (self.target[0] - 0.0040):
          reward += 0.25  
        elif (self.target[0] - 0.0040) > self.pose.position.x > (self.target[0] - 0.0065):
          reward += 0.1
        elif self.pose.position.x < (self.target[0] - 0.0065) or self.pose.position.x > (self.target[0] + 0.0065):
          reward -= 1
        if self.pose.position.x < (self.target[0] - 0.0085) or self.pose.position.x > (self.target[0] + 0.0085):
          reward -= 3
          done = True

        if (self.target[1] + 0.0012) > self.pose.position.y > (self.target[1] - 0.0012):
          reward += 0.5
        elif (self.target[1] + 0.0040) > self.pose.position.y > (self.target[1] + 0.0012):
          reward += 0.25
        elif (self.target[1] + 0.0065) > self.pose.position.y > (self.target[1] + 0.0040):
          reward += 0.1
        elif (self.target[1] - 0.0012) > self.pose.position.y > (self.target[1] - 0.0040):
          reward += 0.25  
        elif (self.target[1] - 0.0040) > self.pose.position.y > (self.target[1] - 0.0065):
          reward += 0.1
        elif self.pose.position.y < (self.target[1] - 0.0065) or self.pose.position.y > (self.target[1] + 0.0065):
          reward -= 1
        if self.pose.position.y < (self.target[1] - 0.0085) or self.pose.position.y > (self.target[1] + 0.0085):
          reward -= 3
          done = True

        # if self.pose.position.z < (self.target[2] + 0.025):
        #   reward += 1.0
        #   if self.pose.position.z < (self.target[2] + 0.015):
        #     reward -= 2
        #     done = True
        # elif (self.target[2] + 0.025) < self.pose.position.z < (self.target[2] + 0.035):
        #   reward += 0.8
        # elif (self.target[2] + 0.035) < self.pose.position.z < (self.target[2] + 0.050):
        #   reward += 0.6
        # elif (self.target[2] + 0.050) < self.pose.position.z < (self.target[2] + 0.065):
        #   reward += 0.4
        # elif (self.target[2] + 0.065) < self.pose.position.z < (self.target[2] + 0.075):
        #   reward += 0.2
        # elif self.pose.position.z > (self.target[2] + 0.080):
        #   reward -= 2


    # if abs(self.pose.position.y - self.target[1]) < abs(self.pose_last.position.y - self.target[1]) :
    #   reward += 0.5
    # elif abs(self.pose.position.y - self.target[1]) > abs(self.pose_last.position.y - self.target[1]) :
    #   reward -= abs(self.pose.position.y - self.target[1])
    # elif self.pose.position.y < (self.target[1] - 0.0005) or self.pose.position.y > (self.target[1] + 0.0015):
    #   reward -= 2

    # if reward < -1:
    #   done = True

    # else:
      # raise ValueError("Received invalid action={} which is not part of the action space".format(action))
    
    self.distance = self.robot.Cal_distance_to_target_and_Force_feedback(self.target)
    # if force_flag == 0:
    self.observation = self.Dis_plus_ForceFeedback(self.distance, self.pose, self.force[0], self.force[1], self.force[2])
    # elif force_flag == 1:
    #   self.observation = self.Dis_plus_ForceFeedback(self.distance, self.force_again[0], self.force_again[1], self.force_again[2])
    # print(self.observation)
    
    self.pose_last = self.pose
    # Optionally we can pass additional info, we are not using that for now
    info = {}
    print("epsoide",self.count,"   Reward:" + str(reward))
    # return np.array([self.agent_pos]).astype(np.float32), reward, done, info
    return np.array([self.observation]).astype(np.float32),  reward, done, info

  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # pose_Gauss_ready = self.Gauss()
    self.robot.go_to_joint_state()
    print(self.hole)

    ############ Real #############
    # self.robot.go_to_pose_goal_Real() 
     
    self.pose_last = self.robot.catch_current_pose()
    if abs(self.pose_last.position.x - self.position_ready[0]) > 0.002 or abs(self.pose_last.position.y - self.position_ready[1]) > 0.002 or abs(self.pose_last.position.z - self.position_ready[2]) > 0.002 :
      self.robot.go_to_joint_state()
      # self.robot.go_to_pose_goal_Real() 
      print("####################### Again ########################")
    self.distance = self.robot.Cal_distance_to_target_and_Force_feedback(self.target)
    self.pose_last = self.robot.catch_current_pose()
    self.observation = self.Dis_plus_ForceFeedback(self.distance, self.pose_last, self.ft_sensor_array[0], self.ft_sensor_array[1], self.ft_sensor_array[2])
    # print(self.target)

    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    # return np.array([self.obs]).astype(np.float32)
    return np.array([self.observation]).astype(np.float32)

  def render(self, mode='human'):
    if mode != 'human':
      raise NotImplementedError()
    pass

  def close(self):
    pass


# rospy.init_node("train_ppo2", anonymous=True, log_level=rospy.INFO)
# env_id = "ur5e-v0"

# log_dir = "/home/ben/work/ur5e_RL/src/RL_env/results/"+env_id+"/"
# os.makedirs(log_dir, exist_ok=True)

# env = gym.make(env_id)
# # env = make_vec_env(env_id, n_envs=4)
# env = Monitor(env, filename=log_dir, allow_early_resets=True)
# # env = PIHAssembleEnv()
# env = DummyVecEnv([lambda: env])

# env.close()

# if self.pose.position.x > (self.target[0] + 0.001):
    #   reward += 1.0
    #   if self.pose.position.x > (self.target[0] + 0.006):
    #     reward -= 3
    #     done = True
    # elif (self.target[0] + 0.001) > self.pose.position.x > (self.target[0] - 0.002):
    #   reward += 0.8
    # elif (self.target[0] - 0.002) > self.pose.position.x > (self.target[0] - 0.004):
    #   reward += 0.5
    # elif (self.target[0] - 0.004) > self.pose.position.x > (self.target[0] - 0.009):
    #   reward += 0.2
    # elif self.pose.position.x < (self.target[0] - 0.014):
    #   reward -= 1
    # elif self.pose.position.x < (self.target[0] - 0.016):
    #   reward -= 3
    #   done = True

    # if (self.target[1]) < self.pose.position.y < (self.target[1] + 0.0005):
    #   reward += 0.5
    #   if self.pose.position.y < (self.target[1] - 0.0005):
    #     reward -= 2
    # elif (self.target[1] + 0.0005) < self.pose.position.y < (self.target[1] + 0.001):
    #   reward += 0.3
    # elif (self.target[1] + 0.0010) < self.pose.position.y < (self.target[1] + 0.0015):
    #   reward += 0.1
    # elif self.pose.position.y > (self.target[1] + 0.0015):
    #   reward -= 2

    # if self.pose.position.z < (self.target[2] + 0.025):
    #   reward += 1.0
    #   if self.pose.position.z < (self.target[2] + 0.020):
    #     reward -= 2
    #     done = True
    # elif (self.target[2] + 0.025) < self.pose.position.z < (self.target[2] + 0.045):
    #   reward += 0.8
    # elif (self.target[2] + 0.045) < self.pose.position.z < (self.target[2] + 0.065):
    #   reward += 0.6
    # elif (self.target[2] + 0.065) < self.pose.position.z < (self.target[2] + 0.085):
    #   reward += 0.4
    # elif (self.target[2] + 0.085) < self.pose.position.z < (self.target[2] + 0.105):
    #   reward += 0.2
    # elif self.pose.position.z > (self.target[2] + 0.125):
    #   reward -= 2



    # if  self.reward_distance < self.reward_distance_last :
    #   reward += (self.reward_distance_last - self.reward_distance) * 2
    # elif self.reward_distance > self.reward_distance_last :
    #   reward -= (self.reward_distance - self.reward_distance_last)
    # elif self.pose.position.x > (self.target[0] + 0.006):
    #   reward -= 3
    #   done = True
    # elif self.pose.position.x < (self.target[0] - 0.016):
    #   reward -= 3
    #   done = True
    # elif self.pose.position.y < (self.target[1] - 0.0005):
    #   reward -= 2
    # elif self.pose.position.y > (self.target[1] + 0.0015):
    #   reward -= 2
    # elif self.pose.position.z < (self.target[2] + 0.018):
    #   reward -= 2
    #   done = True
    # elif self.pose.position.z > (self.target[2] + 0.125):
    #   reward -= 2


    # if self.force_again[0] > 10 or self.force_again[0] < -10 :
    #   print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! break !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #   reward -= 3
    #   force_flag = 1
    #   self.reset()
    # if self.force_again[1] > 20 or self.force_again[1] < -20 :
    #   reward -= 3
    #   force_flag = 1
    #   done = True
    # if self.force_again[2] > 10 or self.force_again[2] < -10 :
    #   reward -= 3
    #   force_flag = 1
    #   done = True
      