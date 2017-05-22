#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 03:23:33 2017

@author: alex
"""
#==============================================================================
# *This is used with train_gpu.py
# Collect the data for training
# For the collaborator we use three types of agents: Focused, Random, Bad
# For our agent we use four types of agent to produce actions: BayesAgent, Random, Bad and Focused
#==============================================================================
from BayesAgent import BayesAgent
from BadAgent import BadAgent
import tensorflow as tf
import os
import sys
sys.path.append('../')
from argparse import ArgumentParser
from datetime import datetime
from os import path
from threading import Thread, active_count, Lock
from time import sleep
try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer

from common import parse_clients_args, visualize_training, ENV_AGENT_NAMES, ENV_TARGET_NAMES
from malmopy.agent import RandomAgent
from agent import PigChaseChallengeAgent, FocusedAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

BASELINES_FOLDER = 'results/bayes_agent/pig_chase/%s/%s'
EPOCH_SIZE = 100


def agent_factory(name, role, type, clients, max_epochs, logdir, visualizer):

    assert len(clients) >= 2, 'Not enough clients (need at least 2)'
    clients = parse_clients_args(clients)
    
    builder = PigChaseSymbolicStateBuilder()
    env = PigChaseEnvironment(clients, builder, role=role,
                              randomize_positions=True)

    if role == 0:
        agent1 = FocusedAgent(name, ENV_TARGET_NAMES[0])
        agent2 = RandomAgent(name, env.available_actions)
        agent3 = BadAgent(name)
        
        agent_list = [agent1, agent2, agent3]# three types of agent
        agent = agent1
        
        obs = env.reset()
        reward = 0
        agent_done = False
        max_training_steps = EPOCH_SIZE * max_epochs
        epoch = 0
        for step in range(1, max_training_steps+1):
            if env.done:
                obs = env.reset()
                epoch += 1
                agent = agent_list[epoch/10 % 3]# change for every 10 episodes
            # select an action
            action = agent.act(obs, reward, agent_done)
            # take a step
            obs, reward, agent_done = env.do(action)
                


    else:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            agent1 = BayesAgent(name, ENV_TARGET_NAMES[0], 'Agent_1', True, sess)
            agent2 = RandomAgent(name, env.available_actions)
            agent3 = BadAgent(name)
            agent4 = FocusedAgent(name, ENV_TARGET_NAMES[0])
            if not agent1.save:
                sess.run(tf.global_variables_initializer()) 
                print "Initialize"
            agent_list = [agent1, agent2, agent3, agent4]# three types of agents
            agent = agent1
            obs = env.reset()
            agent1.reset(obs)
            reward = 0
            agent_done = False
            viz_rewards = []
            avg = []
            epoch = 0
            s = 1
            max_training_steps = EPOCH_SIZE * max_epochs
            for step in range(1, max_training_steps+1):
                # check if env needs reset
                if agent_done:
                    obs = env.reset()
                    agent1.reset(obs)
                    avg.append(sum(viz_rewards))
                    print "Epoch:%d, accumulative rewards: %d"%(epoch, sum(viz_rewards))
                    visualize_training(visualizer, step, viz_rewards)
                    viz_rewards = []
                    epoch += 1
                    agent = agent_list[epoch/5 % 4]# change for every episodes
                    if epoch%10 == 0:
                        agent1.reset_collaborator()
                    s = 1
                    
                # select an action
                action = agent.act(obs, reward, agent_done, is_training = True)
                # take a step
                next_obs, reward, agent_done = env.do(action)
                agent1.collecting(obs, action, reward, next_obs, agent_done, s)
                s += 1
                obs = next_obs
                viz_rewards.append(reward)
                    
                if step % 100 == 0:
                    agent1.save_replay_buffer()
                #
                agent1.inject_summaries(step)
                
            print "Average Reward: ", 1.*sum(avg)/len(avg)

def run_experiment(agents_def):
    assert len(agents_def) == 2, 'Not enough agents (required: 2, got: %d)'\
                % len(agents_def)

    processes = []
    for agent in agents_def:
        p = Thread(target=agent_factory, kwargs=agent)
        p.daemon = True
        p.start()

        # Give the server time to start
        if agent['role'] == 0:
            sleep(1)

        processes.append(p)

    try:
        # wait until only the challenge agent is left
        while active_count() > 2:
            sleep(0.1)
    except KeyboardInterrupt:
        print('Caught control-c - shutting down.')


if __name__ == '__main__':
    arg_parser = ArgumentParser('Pig Chase 6.0 experiment')
    arg_parser.add_argument('-t', '--type', type=str, default='astar',
                            choices=['astar', 'random'],
                            help='The type of baseline to run.')
    arg_parser.add_argument('-e', '--epochs', type=int, default=1000,
                            help='Number of epochs to run.')
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Minecraft clients endpoints (ip(:port)?)+')
    args = arg_parser.parse_args()

    logdir = BASELINES_FOLDER % ('bayes_agent', datetime.utcnow().isoformat())
    if 'malmopy.visualization.tensorboard' in sys.modules:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)
    else:
        visualizer = ConsoleVisualizer()

    agents = [{'name': agent, 'role': role, 'type': args.type,
               'clients': args.clients, 'max_epochs': args.epochs,
               'logdir': logdir, 'visualizer': visualizer}
              for role, agent in enumerate(ENV_AGENT_NAMES)]

    run_experiment(agents)
