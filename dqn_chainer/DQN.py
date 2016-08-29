# -*- coding: utf-8 -*-
"""
Deep Q-network implementation with chainer
Copyright (c) 2016 Naoki Tomii All Right Reserved.
"""

import copy
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

class DQN(chainer.Chain):

  def __init__(self, numAction=10, imgSize=128, depthList=[1, 32, 64, 64], linearSize=512):
    assert imgSize > 0
    assert numAction > 0
    assert len(depthList) is 4
    assert linearSize > 0
    flatLength = ((((((imgSize-8)/4+1)-4)/2+1)-3)/1+1)**2*64
    super(DQN, self).__init__(
        l1=L.Convolution2D(depthList[0], depthList[1], ksize=8, stride=4, wscale=np.sqrt(2)).to_gpu(),
        l2=L.Convolution2D(depthList[1], depthList[2], ksize=4, stride=2, wscale=np.sqrt(2)).to_gpu(),
        l3=L.Convolution2D(depthList[2], depthList[3], ksize=3, stride=1, wscale=np.sqrt(2)).to_gpu(),
        l4=L.Linear(flatLength, linearSize, wscale=np.sqrt(2)).to_gpu(),
        q_value=L.Linear(
          512, numAction,
          initialW=np.zeros((numAction, 512),dtype=np.float32)
        ).to_gpu()
    )

  def __call__(self, x):
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    h3 = F.relu(self.l3(h2))
    h4 = F.relu(self.l4(h3))
    Q = self.q_value(h4)
    return Q

  def reset_state(self):
    self.l1.reset_state()
    self.l2.reset_state()
    self.l3.reset_state()
    self.l4.reset_state()
    self.q_value.reset_state()
  

