# -*- coding: utf-8 -*-
"""
Deep Q-network implementation with chainer and rlglue
Copyright (c) 2015 Naoto Yoshida All Right Reserved.
"""

import copy
import pickle
import numpy as np
import cv2
import sys
import matplotlib
import matplotlib.pyplot as plt

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action

from surface_indices import *

from DQN import DQN_class


class dqn_agent(Agent):  # RL-glue Process
    # parameters
    obsSize = 17
    obsMin = -90.0
    obsMax = 40.0
    imgSize = 128
    imgDepth = 1 
    epsilon = 1.0  
    enable_controller=np.arange(10)
    state = np.zeros((imgDepth, imgSize, imgSize), dtype=np.float32)
    lastAction = Action()
    policyFrozen = False
    cnt = 0

    def obs2state(self, observation):
      self.cnt+=1
      tmp = np.zeros((obsSize,obsSize), dtype=np.float32)
      for i, v in enumerate(observation.doubleArray[:obsSize*obsSize]):
          tmp[surface_indices[i]] = v
      tmp = (tmp - self.obsMin) / (self.obsMax - self.obsMin) # Normalization 
      tmp = cv2.resize(tmp, (self.imgSize, self.imgSize))
      cv2.imshow('observation',tmp)
      cv2.waitKey(100)
      self.state[0] = tmp

    def agent_init(self, taskSpec):
      self.lastAction = Action()
      self.state = np.zeros((self.imgDepth, self.imgSize, self.imgSize), dtype=np.float32)
      self.time = 0
      self.DQN = DQN_class(
              enable_controller=self.enable_controller, 
              imgSize=self.imgSize, imgDepth=self.imgDepth )
      print 'agent_init done'

    def agent_start(self, observation):
      self.obs2state(observation)
      
      returnAction = Action()
      #state_ = cuda.to_gpu(np.asanyarray(
      #    self.state.reshape(1, self.imgDepth, self.imgSize, self.imgSize), 
      #    dtype=np.float32))
      #action, Q_now = self.DQN.e_greedy(state_, self.epsilon)
      returnAction.intArray = [-1]
      
      self.lastAction = copy.deepcopy(returnAction)
      self.last_state = self.state.copy()
      return returnAction

    def agent_step(self, reward, observation):
      self.obs2state(observation)
     
      returnAction = Action()
      actNum = -1
      k = cv2.waitKey(0)
      if k in [ord(v) for v in '123456789']:
        actNum = int(unichr(k)) - 1
      if k == 27:
        raise NameError('Escape')
      returnAction.intArray = [actNum]
      
      self.lastAction = copy.deepcopy(returnAction)
      self.last_state = self.state.copy()
      return returnAction

    def agent_end(self, reward):  # Episode Terminated
      pass

    def agent_cleanup(self):
      pass

    def agent_message(self, inMessage):

      if inMessage.startswith("what is your name?"):
        return "my name is skeleton_agent!"

      if inMessage.startswith("freeze learning"):
          self.policyFrozen = True
          return "message understood, policy frozen"

      if inMessage.startswith("unfreeze learning"):
          self.policyFrozen = False
          return "message understood, policy unfrozen"

      if inMessage.startswith("save model"):
          with open('dqn_model.dat', 'w') as f:
              pickle.dump(self.DQN.model, f)
          return "message understood, model saved"

      if inMessage.startswith("load model"):
          with open('dqn_model.dat', 'r') as f:
              self.DQN.model = pickle.load(f)
          return "message understood, model loaded"

if __name__ == "__main__":
    #AgentLoader.loadAgent(dqn_agent(), "192.168.36.53")
    envIP = '127.0.0.1'
    if len(sys.argv) >= 2:
      envIP = sys.argv[1]
    print 'connecting ' + envIP
    try:
      #AgentLoader.loadAgent(dqn_agent(), envIP)
      AgentLoader.loadAgent(dqn_agent())
    finally:
      cv2.destroyAllWindows()
