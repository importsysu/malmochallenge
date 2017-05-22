#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 03:25:33 2017

@author: alex
"""
#==============================================================================
# This is the algorithm described in the documentation
# But because of the our own problems, we have to split up the process 
# and use another offline version 
# see train_gpu.py
#==============================================================================
import sys, os
sys.path.append('../')
sys.path.append('/home/share/minghan/malmo-challenge/')
sys.path.append('/home/share/minghan/keras/lib/python2.7/site-packages')
import msgpack
import tensorflow as tf
import numpy as np
import random
from malmopy.agent import BaseAgent
from collections import deque, namedtuple
from common import ENV_ACTIONS
from heapq import heapify, heappop, heappush

BATCH_SIZE = 32# default batch size
MEM_SIZE = 1000000# the size of replay buffer
REPLAY_START = 10000
GAMMA = 0.99# discounting factor
EPSILON = 0.25# the probability of agent tacking random action
GOAL_DIMS = 27# the dimensions of the location of the goal, which are the central 25 blocks plus the 2 lapis block 
STATE_DIMS = 9# the dimensions of the state feature vector
class BayesAgent(BaseAgent):
    def __init__(self, name, target, collaborator, is_training, sess, visualizer=None):
        self.actions = [0, 1, 2]#the available actions
        self.goal = np.array([[0,0]])# the location of the goal
        self.obs = None # store the last obs, use to update the collaborator
        self.collaborator = 20*np.ones((1,3))#encode the collaborator's behavior
        self.c_prob = 1.*self.collaborator/(np.sum(self.collaborator, axis = 1)[0])#normalized collaborator vector
        
        super(BayesAgent, self).__init__(str(name), len(self.actions), visualizer)
        self._target = str(target)
        self._collaborator = str(collaborator)
        
        self.sess = sess
        self.state_dims = STATE_DIMS
        self.goal_dims = GOAL_DIMS
        self.replay_buffer = deque()
        self.restore_replay_buffer(is_training)    
        self.state_buffer = []# use to temporarily store the state
        
        self.actor = Actor(str(name))
        self.critic = Critic(self.state_dims, self.goal_dims, sess)
        
        self.save = self.critic.save# check if there's a saved model        
    
    def reset(self, obs):
        self.state_buffer = []
        self.collaborator = self.particle_filtering() # resampling
        self.c_prob = 1.*self.collaborator/(np.sum(self.collaborator, axis = 1)[0])
        print "Replay buffer length:", len(self.replay_buffer)
    

    def act(self, obs, reward, done, is_training = False):
        state = self.state_shaping(obs)
        a_state = np.concatenate([state, self.c_prob], 1)
        self.update_collaborator(obs, reward)
        
        if is_training:
            self.goal = np.array([[state[0][3], state[0][4]]])
            rand = random.random()
            if rand > EPSILON:#epsilon greedy 
                action = self.actor.get_action(obs, reward, done, self.goal, is_training)
            else:
                action = random.choice([0,1,2])
        else:
            if self.c_prob[0][2] > 0.5:# use particle filter
                if self.manhattan_dist((state[0][0], state[0][1]), (1,4)) > self.manhattan_dist((state[0][0], state[0][1]), (7,4)):
                    return self.actor.get_action(obs, reward, done, [[7,4]], is_training)
                else:
                    return self.actor.get_action(obs, reward, done, [[1,4]], is_training)
            me = self.actor.Neighbour(1, state[0][0], state[0][1], 0, "")
            target = self.actor.Neighbour(1, state[0][2], state[0][3], 0, "")
            path, _ = self.actor._find_shortest_path(me, target, state=obs[0])
            if 2 <= len(path) <= 4:
                n = path[-2]
                if self.manhattan_dist((int(state[0][2]), int(state[0][3])), (n.x,n.z)) == 1 \
                           and self.manhattan_dist((int(state[0][4]), int(state[0][5])), (n.x,n.z)) == 2:
                    action = self.actor.get_action(obs, reward, done, [[n.x, n.z]], is_training)
                    #print "Action:", action
                    return action
            self.goal = self.goal_shaping(self.critic.get_goal(a_state))
            action = self.actor.get_action(obs, reward, done, self.goal, is_training)
            
        return action

    def training(self, obs, action, reward, next_obs, done, step, s):
        
        state = self.state_shaping(obs)
        next_state = self.state_shaping(next_obs)
        r = self.reward_shaping(next_state, reward)
        
        a_state = np.concatenate([state, self.c_prob], 1)
        a_next_state = np.concatenate([next_state, self.c_prob], 1)
        # Goal Swapping
        goal_swap = self.reverse_goal_shaping(np.array([[next_state[0][0], next_state[0][1]]]))#use the current location as the goal
        self.state_buffer.append(a_state)
        for i in range(step): 
            r_sum = 0
            for j in range(0, step-i-1):
                r_sum += -1.0*(GAMMA**j) # summing the discounted reward from the start to the current state for every state in the state buffer
            self.store_transition([self.state_buffer[i].tolist(), goal_swap, \
                              r_sum+(GAMMA**(step-i-1))*r, a_next_state.tolist(), GAMMA**step])
        
        if len(self.replay_buffer) > REPLAY_START:
            batch = self.sampling()
            l = self.update_model(batch)
            if s%1 == 0:
                print "Epoch:%d, loss: %d"%(s, l)
            
            if s % 1000 == 0:# update the target network every 10000 steps
                print "Update Target Network"
                self.critic.update_target()
            
            if s % 50 == 0:
                self.critic.save_model()
            
            
    
    def matches(self, a, b):
        return a.x == b.x and a.z == b.z # victory!!
    
    
    #==============================================================================
    # Particle filter: update beliefs
    # Collaborator's Update Method
    # we categorize the type of agent as three classes: [Focused, Random, Bad], in which Bad means tends to move to the lapis block
    # and at every step, we increase the corresponding one based on the collaborator's behavior
    # It's hard to specify what's the behavior of an agent
    # so we do a lot of handcraft to specify the rules
    # it's noisy but still can be a prior
    # for example, a focused agent might look like [ 27, 15, 8 ]
    #==============================================================================
    def reset_collaborator(self):# the particle filter is very sensitive to initial value, so we set it to [20, 20, 20] to change slowly over time
        self.collaborator = 20*np.ones((1,3))
        self.c_prob = 1.*self.collaborator/(np.sum(self.collaborator, axis = 1)[0])
        
    def update_collaborator(self, next_obs, reward):
        if self.obs is None:
            self.obs = next_obs
            return
        state = self.state_shaping(self.obs)
        next_state = self.state_shaping(next_obs)
        
        target_pos = (state[0][2], state[0][3])
        collaborator_pos = (state[0][4], state[0][5])
        new_collaborator_pos = (next_state[0][4], next_state[0][5])
        
        ori_pig_dist = self.manhattan_dist(target_pos, collaborator_pos)# distance between pig and collaborator at last state
        new_pig_dist = self.manhattan_dist(target_pos, new_collaborator_pos)# distance between pig and collaborator at this state
        
        ori_leave1_dist = self.manhattan_dist((1, 4), collaborator_pos)# distance between lapis blocks and collaborator 
        new_leave1_dist = self.manhattan_dist((1, 4), new_collaborator_pos)
        
        ori_leave2_dist = self.manhattan_dist((7, 4), collaborator_pos)
        new_leave2_dist = self.manhattan_dist((7, 4), new_collaborator_pos)
        
        #complicated rules, did a lot of handcrafting and tuning
        if ori_leave1_dist < ori_leave2_dist and ori_pig_dist < new_pig_dist:
            if ori_leave1_dist >= new_leave1_dist:
                self.collaborator[0][2] += 1
            elif ori_leave1_dist <= new_leave1_dist:
                self.collaborator[0][1] += 1
                                 
        elif ori_leave1_dist > ori_leave2_dist and ori_pig_dist < new_pig_dist:
            if ori_leave2_dist >= new_leave2_dist:
                self.collaborator[0][2] += 1
            elif ori_leave2_dist <= new_leave2_dist:
                self.collaborator[0][1] += 1
        
        elif ori_pig_dist > new_pig_dist:
            self.collaborator[0][0] += 1
                             
        if new_collaborator_pos == (1, 4) or new_collaborator_pos == (7, 4):
            self.collaborator[0][2] += 10
        #normalize the collaborator behavior vector to get probability
        self.c_prob = 1.*self.collaborator/(np.sum(self.collaborator, axis = 1)[0])
        self.obs = next_obs
        
    #==============================================================================
    # Particle filter: Resampling
    # making it more easier for our agent to adapt to the changes of the collaborator
    #==============================================================================
    def particle_filtering(self):# the particle filter tends to be unstable if samples are not enough
        temp = np.zeros((1,3))
        #print self.c_prob[0]
        for i in range(50):
            index = np.random.choice([0,1,2], 1, p=self.c_prob[0])[0]
            temp[0][index] += 1
        print "Collaborator: ", temp
        return temp
    #==============================================================================
    
    def update_model(self, batch):
        start_batch = batch[0]
        goal_batch = batch[1]
        reward_batch = batch[2]
        end_batch = batch[3]
        step_batch = batch[4]
        
        result = self.critic.update(start_batch, goal_batch, reward_batch, end_batch, step_batch)
        return result
    
    def save_replay_buffer(self):
        with open('replay_buffer.msg','w') as f:
            f.truncate()
            replay_buffer = {'replay_buffer': list(self.replay_buffer)}
            print "Replay Buffer Saved"
            msgpack.pack(replay_buffer, f)
        
    def restore_replay_buffer(self, is_training):
        if is_training:
            if os.path.exists('replay_buffer.msg'):
                with open('replay_buffer.msg', 'r') as f:
                    data = msgpack.unpack(f)
                    print len(data['replay_buffer'])
                    print "Loading Replay Buffer"
                    self.replay_buffer = deque(data['replay_buffer'])
            else:
                print "Initialize Replay Buffer"
                self.replay_buffer = deque()

    def store_transition(self, sample):#store the transition in the replay buffer
        self.replay_buffer.append(sample)
        if len(self.replay_buffer) > MEM_SIZE:
            self.replay_buffer.popleft()
        
        return len(self.replay_buffer)
    
    def sampling(self):# sample minibatch from the experience replay buffer
        if len(self.replay_buffer) >= BATCH_SIZE:
            batch = random.sample(self.replay_buffer,BATCH_SIZE)
            start_batch = []
            goal_batch = []
            reward_batch = []
            end_batch = []
            step_batch = []
            
            for data in batch:
                start_batch.append(data[0]) 
                goal = np.zeros((1, GOAL_DIMS))
                goal[0,data[1]] = 1
                goal_batch.append(goal)
                reward_batch.append(data[2])
                end_batch.append(data[3])
                step_batch.append(data[4])

            return [np.array(start_batch).reshape(BATCH_SIZE, STATE_DIMS), np.array(goal_batch).reshape(BATCH_SIZE, GOAL_DIMS), \
                    np.array(reward_batch).reshape(BATCH_SIZE, 1), np.array(end_batch).reshape(BATCH_SIZE, STATE_DIMS),\
                    np.array(step_batch).reshape(BATCH_SIZE, 1)]
    
    
    def state_shaping(self, obs):#flatten the state to input shape, modified from the sample code of FocusedAgent 
        state = obs[0]
        #My agent position
        me = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k]
        #target position
        target = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self._target in k]
        #opponet postion
        collaborator = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self._collaborator in k] 
        #flatten
        if me is [] or target is [] or collaborator is []:
            me = [(j, i) for i, v in enumerate(self.obs[0]) for j, k in enumerate(v) if self.name in k]
            #target position
            target = [(j, i) for i, v in enumerate(self.obs[0]) for j, k in enumerate(v) if self._target in k]
            #opponet postion
            collaborator = [(j, i) for i, v in enumerate(self.obs[0]) for j, k in enumerate(v) if self._collaborator in k]

        inputs = np.array([[me[0][0], me[0][1], target[0][0], target[0][1], \
                                collaborator[0][0], collaborator[0][1]]])
        return inputs
    
    def goal_shaping(self, goal):# shape the index of goal to the coordinate form
        if goal < 25:
            x = goal % 5 + 2
            z = goal / 5 + 2
        else:
            if goal == 25:
                x = 1
                z = 4
            if goal == 26:
                x = 7
                z = 4
        return np.array([[x,z]])
    
    def reverse_goal_shaping(self, pos):# shape the coordinate of goal to the index form
        goal = (pos[0][1]-2)*5 +(pos[0][0]-2) 
        if goal < 25:
            return goal
        elif (pos[0][0], pos[0][1]) is (1,4):
            return 25
        elif (pos[0][0], pos[0][1]) is (7,4):
            return 26
    
    def reward_shaping(self, state, reward):# shape the reward
        collaborator = (state[0][4], state[0][5])
        target = (state[0][2], state[0][3])
        if reward > 10: # if the collaborator catch the pig in the lapis block, then our agent will not be rewarded
            if collaborator == (2,4) and target == (1, 4):
                reward = 0
            elif collaborator == (6,4) and target == (7, 4):
                reward = 0

        return reward
    
    def manhattan_dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
class Critic(object):
    def __init__(self, state_dims, goal_dims, sess):
        self.sess = sess
        self.state_dims = state_dims
        self.goal_dims = goal_dims
        self.build_model()
        self.new_pars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='new_network')
        self.tar_pars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
            self.save = True
        else:
            print "Could not find old Critic network weights"   
            self.save = False
            
    def build_model(self):# 4 layres, each layer has 1024 neurons with rectifier nonlinearity
        def weight_variable(shape):
            W = tf.get_variable('W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            return W
    
    	def bias_variable(shape):
            initial = tf.constant(0.01, shape = shape)
            return tf.Variable(initial)
        
        with tf.variable_scope('inputs'):
                self.state_input = tf.placeholder('float32', [None, self.state_dims])
                self.y_input = tf.placeholder('float32', [None, ])
                self.goal_input = tf.placeholder('float32', [None, self.goal_dims])
                
        with tf.variable_scope('new_network'):
            with tf.variable_scope('layer_1'):
                self.W_fc1 = weight_variable([self.state_dims,1024])
                self.b_fc1 = bias_variable([1024])
                self.h_fc1 = tf.nn.relu(tf.matmul(self.state_input,self.W_fc1) + self.b_fc1)

            with tf.variable_scope('layer_2'):
                self.W_fc2 = weight_variable([1024,1024])
                self.b_fc2 = bias_variable([1024])
                self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1,self.W_fc2) + self.b_fc2)

            with tf.variable_scope('layer_3'):
                self.W_fc3 = weight_variable([1024,1024])
                self.b_fc3 = bias_variable([1024])
                self.h_fc3 = tf.nn.relu(tf.matmul(self.h_fc2,self.W_fc3) + self.b_fc3)

            with tf.variable_scope('layer_4'):
                self.W_fc4 = weight_variable([1024,1024])
                self.b_fc4 = bias_variable([1024])
                self.h_fc4 = tf.nn.relu(tf.matmul(self.h_fc3,self.W_fc4) + self.b_fc4)

            with tf.variable_scope('layer_5'):
                self.W_fc5 = weight_variable([1024,self.goal_dims])
                self.b_fc5 = bias_variable([self.goal_dims])
                self.qvalue = tf.matmul(self.h_fc4,self.W_fc5) +self.b_fc5
                self.prediction = tf.reduce_sum(self.qvalue * self.goal_input, axis=1)

                
        with tf.variable_scope('target_network'):
            with tf.variable_scope('layer_1'):
                self._W_fc1 = weight_variable([self.state_dims,1024])
                self._b_fc1 = bias_variable([1024])
                self._h_fc1 = tf.nn.relu(tf.matmul(self.state_input,self._W_fc1) + self._b_fc1)

            
            with tf.variable_scope('layer_2'):
                self._W_fc2 = weight_variable([1024,1024])
                self._b_fc2 = bias_variable([1024])
                self._h_fc2 = tf.nn.relu(tf.matmul(self._h_fc1,self._W_fc2) + self._b_fc2)

                
            with tf.variable_scope('layer_3'):
                self._W_fc3 = weight_variable([1024,1024])
                self._b_fc3 = bias_variable([1024])
                self._h_fc3 = tf.nn.relu(tf.matmul(self._h_fc2,self._W_fc3) + self._b_fc3)

            with tf.variable_scope('layer_4'):
                self._W_fc4 = weight_variable([1024,1024])
                self._b_fc4 = bias_variable([1024])
                self._h_fc4 = tf.nn.relu(tf.matmul(self._h_fc3,self._W_fc4) + self._b_fc4)

            with tf.variable_scope('layer_5'):
                self._W_fc5 = weight_variable([1024,self.goal_dims])
                self._b_fc5 = bias_variable([self.goal_dims])
                self._qvalue = tf.matmul(self._h_fc4,self._W_fc5) +self._b_fc5
                self._prediction = tf.reduce_sum(self._qvalue * self.goal_input, axis=1)

                
        with tf.variable_scope('train'):
            self.delta = self.y_input - self.prediction
            self.loss = tf.reduce_mean(tf.square(self.delta))#mean squared error
            self.train_step = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.loss)

        
    def get_qvalue(self, state):
        qvalue = self.sess.run(self.qvalue, feed_dict = {self.state_input: state})
        return qvalue
    
    def get_goal(self, state):
        qvalue = self.sess.run(self.qvalue, feed_dict = {self.state_input: state})
        #print "Goal:", np.argmax(qvalue, axis = 1)
        return np.argmax(qvalue, axis = 1)[0]
        
    def update(self, state, goal, reward, next_state, step):#using double q learning
        nqvalue = self.sess.run(self.qvalue, feed_dict = {self.state_input: next_state})
        ngoal = np.zeros((BATCH_SIZE,self.goal_dims))
        index = np.argmax(nqvalue, axis = 1)
        for i in xrange(BATCH_SIZE):
            ngoal[i][index[i]] = 1
        double_q = self.sess.run(self._prediction, feed_dict = {self.state_input: next_state, self.goal_input: ngoal})
        target = reward[:,0] + step[:,0] * double_q
        train, l, qvalue = self.sess.run([self.train_step, self.loss, self.qvalue], \
                                feed_dict = {self.state_input: state, self.goal_input: goal,\
                                             self.y_input: target})
        return l
    
    def update_target(self):#update the target network every C steps
	self.new_pars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='new_network')
        self.tar_pars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
        self.sess.run([tf.assign(tar, new) for tar, new in zip(self.tar_pars, self.new_pars)])
        
    def restore_model(self):
        self.saver.restore(self.sess, 'saved_critic/critic.ckpt')
        print "Critic Model Restored"
        
    def save_model(self):
        self.saver.save(self.sess, 'saved_critic/critic.ckpt')
        print "Critic Model Saved"
        

class Actor(object):# modified from the samaple code of FousedAgent provided by Malmo
    def __init__(self, name):
        self.name = name
        self._previous_target_pos = None
        self._action_list = []
        self.ACTIONS = ENV_ACTIONS
        self.Neighbour = namedtuple('Neighbour', ['cost', 'x', 'z', 'direction', 'action'])
        
    def get_action(self, state, reward, done, goal, is_training):
        if done:
            self._action_list = []
            self._previous_target_pos = None
        entities = state[1]
        state = state[0]

        me = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k]
        if len(me) < 1:
            return None
        me_details = [e for e in entities if e['name'] == self.name][0]
        yaw = int(me_details['yaw'])
        direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.
        target = [(goal[0][0], goal[0][1])]
        #print "In get action goal:", goal
        # Get agent and target nodes
        me = self.Neighbour(1, me[0][0], me[0][1], direction, "")
        target = self.Neighbour(1, target[0][0], target[0][1], 0, "")
        
        if not self._previous_target_pos == target:
            # Target has moved, or this is the first action of a new mission - calculate a new action list
            self._previous_target_pos = target

            path, costs = self._find_shortest_path(me, target, state=state)
            self._action_list = []
            for point in path:
                self._action_list.append(point.action)

        if self._action_list is not None and len(self._action_list) > 0:
            action = self._action_list.pop(0)
            return self.ACTIONS.index(action)
            #print "Action: ", action

        # reached end of action list - turn on the spot
        return self.ACTIONS.index("turn 1")  # substitutes for a no-op command
    
    def _find_shortest_path(self, start, end, **kwargs):
        came_from, cost_so_far = {}, {}
        explorer = []
        heapify(explorer)

        heappush(explorer, (0, start))
        came_from[start] = None
        cost_so_far[start] = 0
        current = None

        while len(explorer) > 0:
            _, current = heappop(explorer)

            if self.matches(current, end):
                break
            for nb in self.neighbors(current, **kwargs):
                cost = nb.cost if hasattr(nb, "cost") else 1
                new_cost = cost_so_far[current] + cost

                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    priority = new_cost + self.heuristic(end, nb, **kwargs)
                    heappush(explorer, (priority, nb))
                    came_from[nb] = current

        # build path:
        path = deque()
        
        while current is not start:
            path.appendleft(current)
            current = came_from[current]

        return path, cost_so_far
    
    def neighbors(self, pos, state=None):
        state_width = state.shape[1]
        state_height = state.shape[0]
        dir_north, dir_east, dir_south, dir_west = range(4)
        neighbors = []
        inc_x = lambda x, dir, delta: x + delta if dir == dir_east else x - delta if dir == dir_west else x
        inc_z = lambda z, dir, delta: z + delta if dir == dir_south else z - delta if dir == dir_north else z
        # add a neighbour for each potential action; prune out the disallowed states afterwards
        for action in self.ACTIONS:
            if action.startswith("turn"):
                neighbors.append(
                    self.Neighbour(1, pos.x, pos.z, (pos.direction + int(action.split(' ')[1])) % 4, action))
            if action.startswith("move "):  # note the space to distinguish from movemnorth etc
                sign = int(action.split(' ')[1])
                weight = 1 if sign == 1 else 1.5
                neighbors.append(
                    self.Neighbour(weight, inc_x(pos.x, pos.direction, sign), inc_z(pos.z, pos.direction, sign),
                                           pos.direction, action))
            if action == "movenorth":
                neighbors.append(self.Neighbour(1, pos.x, pos.z - 1, pos.direction, action))
            elif action == "moveeast":
                neighbors.append(self.Neighbour(1, pos.x + 1, pos.z, pos.direction, action))
            elif action == "movesouth":
                neighbors.append(self.Neighbour(1, pos.x, pos.z + 1, pos.direction, action))
            elif action == "movewest":
                neighbors.append(self.Neighbour(1, pos.x - 1, pos.z, pos.direction, action))

        # now prune:
        valid_neighbours = [n for n in neighbors if
                            n.x >= 0 and n.x < state_width and n.z >= 0 and n.z < state_height and state[
                                n.z, n.x] != 'sand']
        return valid_neighbours

    def heuristic(self, a, b, state=None):
        (x1, y1) = (a.x, a.z)
        (x2, y2) = (b.x, b.z)
        return abs(x1 - x2) + abs(y1 - y2)
    
    def matches(self, a, b):
        return a.x == b.x and a.z == b.z
