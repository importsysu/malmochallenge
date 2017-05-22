#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 07:27:00 2017

@author: alex
"""
#==============================================================================
# *This is used with collect_data_cpu.py
# when data collecting is done, we train our model on GPUs
#==============================================================================
import sys, os
sys.path.append('../')
sys.path.append('/home/share/minghan/malmo-challenge/')
sys.path.append('/home/share/minghan/keras/lib/python2.7/site-packages')
from BayesAgent_gpu import BayesAgent
import tensorflow as tf

from common import ENV_TARGET_NAMES

EPOCH_SIZE = 10000000

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    agent = BayesAgent('Agent2', ENV_TARGET_NAMES[0], 'Agent_1',True, sess)
    writer = tf.train.SummaryWriter("./logs/malmo", sess.graph)
    if not agent.save:
        sess.run(tf.initialize_all_variables()) 
        print "Initialize"
    for epoch in xrange(EPOCH_SIZE):
        l, summary = agent.training()
        writer.add_summary(summary, epoch)
        if epoch%100 == 0:
            print "Epoch:%d, loss: %d"%(epoch, l)
        
    if epoch % 10000 == 0:# update the target network every 10000 steps
        print "Update Target Network"
        agent.critic.update_target()
    
    if epoch % 1000 == 0:
            agent.critic.save_model()
