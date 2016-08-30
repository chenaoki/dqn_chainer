# -*- coding: utf-8 -*-
"""
Deep Q-network implementation with chainer and rlglue
Copyright (c) 2016 Naoki Tomii All Right Reserved.
"""

import copy
import pickle
import numpy as np
import cv2
import sys
from chainer import cuda, optimizers, serializers, Variable
from chainer import functions as F

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action

from DQN import DQN

class dqn_agent(Agent):  # RL-glue Process
    # Hyper-Parameters
    epsilon = 1.0  
    numAction = 10 
    gamma = 0.99                      # Discount factor
    data_size = 10**5                 # Data size of history. original: 10^6
    initial_exploration = 100         # Initial exploratoin. original: 5x10^4
    replay_size = 32                  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4

    def agent_init(self, taskSpec):
      print "[Age] init ",
      print 'taskSpec : ' + taskSpec,
      self.time = 0
      self.lastAction = None 
      self.policyFrozen = False
      self.imgSize = int(taskSpec)
      self.imgDepth = 1
      self.actions = range(self.numAction)
      self.model = DQN(numAction=self.numAction,imgSize=self.imgSize)
      self.model_t = copy.deepcopy(self.model)
      self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
      self.optimizer.setup(self.model)
      # History Data : D=[s, a, r, s_dash, end_episode_flag]
      self.D =[np.zeros((self.data_size, self.imgDepth,self.imgSize,self.imgSize),dtype=np.float32),
               np.zeros((self.data_size, 1), dtype=np.uint8),
               np.zeros((self.data_size, 1), dtype=np.int8),
               np.zeros((self.data_size, self.imgDepth,self.imgSize,self.imgSize),dtype=np.float32),
               np.zeros((self.data_size, 1), dtype=np.bool)]
      print 'done'

    def agent_start(self, observation):
      print "[Age] start ...",

      self.state = self.obs2state(observation)
      self.time+=1

      action = self.egreedy()

      self.lastAction = copy.deepcopy(action)
      self.lastState = self.state.copy()
      print "done"
      return action

    def agent_step(self, reward, observation):
      print "[Age] step ...",
      print "time {0},".format(self.time),
      
      self.state = self.obs2state(observation)
      self.time+=1

      action = self.egreedy()
      self.updateModel(reward)

      self.lastAction = copy.deepcopy(action)
      self.lastState = self.state.copy()
      print "done"
      return action

    def agent_end(self, reward):  # Episode Terminated
      self.updateModel(reward)

    def agent_cleanup(self):
      pass

    def agent_message(self, inMessage):
      print "[Age] received {0}".format(inMessage)

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
              pickle.dump(self.model, f)
          return "message understood, model saved"

      if inMessage.startswith("load model"):
          with open('dqn_model.dat', 'r') as f:
              self.model = pickle.load(f)
          return "message understood, model loaded"
    
    def obs2state(self, observation):
      retImage = np.array(observation.doubleArray).astype(np.float32)
      retImage = retImage.reshape((self.imgSize, self.imgSize))
      return retImage[np.newaxis, :, :]

    def egreedy(self):
      
      # Exploration decays (eps update)
      if self.policyFrozen is False:  # Learning ON/OFF
        if self.initial_exploration < self.time:
          self.epsilon -= 1.0/10**6
          if self.epsilon < 0.1:
              self.epsilon = 0.1
        else:  # Initial Exploation Phase
          #print "Initial Exploration : ", 
          #print "%d/%d steps" % (self.time, self.initial_exploration)
          self.epsilon = 1.0
      else:  # Evaluation
        self.epsilon = 0.05
      print 'epsilon : {0},'.format( self.epsilon ),
      
      s = Variable(cuda.to_gpu( self.state[np.newaxis,:,:,:] ))
      Q_now = self.model(s).data
      if np.random.rand() < self.epsilon:
        index_action = np.random.randint(0, len(self.actions))
        print "RANDOM,",
      else:
        index_action = np.argmax(Q_now.get())
        print "GREEDY,",
        print Q_now, 
      print 'action : {0},'.format(index_action),
      action = Action(numInts=1)
      action.intArray = [self.actions[index_action]]
      return action 


    def updateModel(self, reward):
      
      if self.policyFrozen is False:

        # Stock experience
        data_index = self.time % self.data_size
        self.D[0][data_index] = self.lastState
        self.D[1][data_index] = self.lastAction.intArray[0]
        self.D[2][data_index] = reward
        self.D[3][data_index] = self.state
        self.D[4][data_index] = False

        # Experience replay
        if self.time > self.initial_exploration:

          # Pick up replay data
          index_max = min(self.data_size, self.time)
          replay_index       = np.random.randint(0, index_max, self.replay_size)
          s_replay           = self.D[0][replay_index]
          a_replay           = self.D[1][replay_index]
          r_replay           = self.D[2][replay_index]
          s_dash_replay      = self.D[3][replay_index]
          episode_end_replay = self.D[4][replay_index]

          # Gradient-based update
          s      = Variable(cuda.to_gpu(s_replay))
          s_dash = Variable(cuda.to_gpu(s_dash_replay))

          Q          = self.model(s)
          tmp        = self.model_t(s_dash)
          tmp        = list(map(np.max, tmp.data.get()))  # max_a Q(s',a)
          max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
          target     = np.asanyarray(Q.data.get(), dtype=np.float32)

          for i in range(self.replay_size):
            if not episode_end_replay[i][0]:
              tmp_ = np.sign(r_replay[i]) + self.gamma * max_Q_dash[i]
            else:
              tmp_ = np.sign(r_replay[i])
            action_index = self.actions.index(a_replay[i])
            target[i, action_index] = tmp_

          # TD-error clipping
          td = Variable(cuda.to_gpu(target)) - Q  # TD error
          td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
          td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

          zero_val = Variable(cuda.to_gpu(np.zeros((self.replay_size, len(self.actions)), dtype=np.float32)))
          loss = F.mean_squared_error(td_clip, zero_val)

          self.optimizer.zero_grads()
          loss.backward()
          self.optimizer.update()

      # Target model update
      if self.time > self.initial_exploration: 
        if np.mod(self.time, self.target_model_update_freq) == 0:
          self.model_t = copy.deepcopy(self.model)

if __name__ == "__main__":
    try:
      AgentLoader.loadAgent(dqn_agent())
    finally:
      cv2.destroyAllWindows()
