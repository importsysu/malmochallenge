# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================
import sys, os
sys.path.append('../')
from common import ENV_AGENT_NAMES, ENV_TARGET_NAMES
from evaluation import PigChaseEvaluator
from environment import PigChaseSymbolicStateBuilder
from malmopy.agent import RandomAgent
from agent import PigChaseChallengeAgent, FocusedAgent
from BayesAgent import BayesAgent
import tensorflow as tf

if __name__ == '__main__':
    # Warn for Agent name !!!
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
    	agent = BayesAgent('Agent_2', ENV_TARGET_NAMES[0], 'Agent_1', False, sess)
        if not agent.save:
            sess.run(tf.global_variables_initializer()) 
            print "Initialize"
        clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
        eval = PigChaseEvaluator(clients, agent, agent, PigChaseSymbolicStateBuilder())
        eval.run()
    
        eval.save('HiDDeN Vs PigChaseChallenger', 'pig_chase_results')
