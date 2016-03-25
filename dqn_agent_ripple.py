# -*- coding: utf-8 -*-
"""
Deep Q-network implementation with chainer and rlglue
Copyright (c) 2015 Naoto Yoshida All Right Reserved.
"""

import copy

import pickle
import numpy as np
import cv2
import scipy.misc as spm
import sys
import matplotlib
import matplotlib.pyplot as plt

from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action

from surface_indices import *

#from dqn_agent_nature import DQN_class

class dqn_agent(Agent):  # RL-glue Process
    lastAction = Action()
    policyFrozen = False
    cnt = 0

    def agent_init(self, taskSpec):
      # Some initializations for rlglue
      self.lastAction = Action()

      self.time = 0
      self.epsilon = 1.0  # Initial exploratoin rate

      # Pick a DQN from DQN_class
      #self.DQN = DQN_class()  # Default is for "Pong".
      
      print 'agent_init done'

    def agent_start(self, observation):
      returnAction = Action()
      returnAction.intArray = [-1]
      return returnAction

    def agent_step(self, reward, observation):
      # Observation display
      returnAction = Action()
      actNum = -1
      #if self.cnt % 100 < 5: 
      #  actNum = (self.cnt // 100) % 9
      img = np.zeros((17,17), dtype=np.float32)
      for i, v in enumerate(observation.doubleArray[:17*17]):
          img[surface_indices[i]] = v
      img = (img + 90.0) / 130.0
      img = cv2.resize(img, (128, 128))
      cv2.imshow('observation',img)
      k = cv2.waitKey(0)
      if k in [ord(v) for v in '123456789']:
        actNum = int(unichr(k)) - 1
      if k == 27:
        raise NameError('Escape')
      returnAction.intArray = [actNum]
      self.cnt+=1
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
      AgentLoader.loadAgent(dqn_agent(), envIP)
    finally:
      cv2.destroyAllWindows()
