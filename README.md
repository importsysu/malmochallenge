# Hierarchical Double Deep Q-Network (HiDDeN)
Minghan Li, Fan Mo, Ancong Wu

----
## Introduction
The Pig Chase Challenge task basically requires us to design an agent in cooperation with another to catch a pig in a fence. Success in catching the pig rewards 25 points. The agent can also choose to go to the lapis blocks to get 5 points and end the game early. The agent will be tested with multiple kinds of cooperators and benchmark its overall performance. Further descriptions of this challenge can be found [here](https://github.com/Microsoft/malmo-challenge/blob/master/ai_challenge/pig_chase/README.md).

![Pig chase](doc/pig-chase-overview.png)
>Figure 1: The overview of the pig chase challenge from https://github.com/Microsoft/malmo-challenge/blob/master/ai_challenge/pig_chase/README.md.


## Installation Dependencies
* Python 2.7
* Tensorflow 1.0
* MessagePack
* Numpy
* [Other Dependencies](https://github.com/Microsoft/malmo-challenge)

## Get Started
* After installing all dependencies, put all files of the `./HiDDeN/` into `malmo-challenge/ai_challenge/pig_chase/`([directory](https://github.com/Microsoft/malmo-challenge/blob/master/ai_challenge/pig_chase/README.md)). 

* Run `python test.py` to see the agent's performance.

* Run `python train.py` if you want to retrain the agent.

* To use GPUs, install tensorflow 0.8 rather than 1.0, and run `python train_gpu.py`.

## Abstract
The difficulty of this challenge comes from the uncertainty of the collaborator’s behaviors and the conditions of success(requires two agents to corner the pig rather than to move directly to the pig's location). Under the context of this task, we hope the agent can learn strategies like flanking and ambushing as well as taking advantage of the behavioral pattern of the collaborator to catch the pig. Therefore, **temporal abstraction** and **inference** are needed for this specific task. We make three main contributions in this work:
1. Combine Double Deep Q-Network [[3]](#reference) with the option framework [[2]](#reference) to produce the Hierarchical Double Deep Q-Network, or **HiDDeN**;
1. Add **particle filter** [[4]](#reference) module to make inference to the collaborator’s behavior;

Because of the our limited resources, we have to split the learning process into data collecting on CPUs and training on GPUs. As we know the model will be likely to overfit the dataset in this way, especially for the model like neural network. However, the result does show that with HiDDeN, the agent is able to learn some high level strategies and emerges collaborative patterns.
### Video presentiation
* Youtube [Link](https://youtu.be/GR5rj8rRy1c)


## Model structure
![Model structure](doc/chart-cut.png)
>Figure 2: The Critic contains a Deep Q-Network[1], outputs Q-values for each goal given the current state. The Meta is the central controller of the hierarchical model, receives Q-values from the critic and specifies the goal. The Actor is an AStar agent, which moves greedily to the current goal. The particle filter module is used to encode the behavior of our collaborator.

The temporal abstractions (high level strategies) are usually very difficult to define, even to be learned [[5]](#reference). Thus, we use the concept of sub goals [[6]](#reference), which are specific coordinates in this task. Therefore the Q-value function we use is Q(s, g) instead of Q(s, a). It avoids training the agent using primitive actions and speeds up the data collecting process.

**Critic Module:**
The Critic uses a fully connected neural network with 4 hidden layers, each layer has 1024 neurons with a rectifier nonlinearity. It takes modified state feature vector and the goal as input, i.e. Q(s, g). To stabilize the training process and break the correlations among the data, we also use experience replay and target network [[1]](#reference) in our model.

Q-learning is known to have the overestimation problem, especially under the non-stationary and stochastic environments. This can be addressed by Double Q learning [[7]](#reference). To incorporate Double Q learning into DQN, we take the method from [[3]](#reference), using the target network to estimate the current network’s Q value. Now we have a new update rule:

![formula](doc/formula-cut.png)

**Meta Module:**
The Meta produces goal based on the Q value from the Critic, and it also uses **particle filter** [[4]](#reference) to update the agent’s belief about the behavioral pattern of the collaborator. We use a vector to encode the collaborator’s type by using the noisy reading from its behavior. By doing resampling from the normalized probability vector in each episode, we can make our agent more adaptive to the changes of the collaborator’s behavior. The normalized vector will also be concatenated with the state feature vector as the input of the Critic.

**Actor Module:**
It basically is an AStar agent, which receives goal from Meta and act greedily to it. The code for the Actor Module is modified from the provided AStar agent [[8]](#reference).

**Off Policy Learning:**
At the data collecting process we don’t use Meta to output a goal, but instead, we use our current coordinate as the goal to update all the previous states within an episode. In this manner, say if our episode is 25 steps long, then we can gather 325(sum 1 to 25) data within one episode. We use three behavior policies to interact with the collaborator: Always chasing the pig, random walking and always going to the lapis block. Since we know that combining TD learning, function approximation and off policy learning will easily cause divergence, we tried different tricks to avoid that. However, that still causes our agent behave strangely and get stuck in some states.

### The HiDDeN Algorithm
Since we dont't deploy Project Malmo on GPUs, the whole learning process is split into two stages: data collecting on CPUs and training on GPUs, aka we will be using the offline version of the algorithm to train our agent.

#### Online HiDDeN Algorithm
![Algorithm](doc/algo-online-cut.png)

#### Offline HiDDeN Algorithm
![Algorithm](doc/algo-offline-cut.png)

## Evaluation Results (compare with the baseline)
![VS focused](doc/results.png)
>Figure 3: The results of HiDDeN vs Focused and Focused vs Focused. The Red Line represents the Fouced agent and the Blue one represents HiDDeN agent. We can see our method indeed outperforms the astar heuristic.

---
## Reference
* [1] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. _Nature, 2015, 518(7540): 529-533._
* [2] Sutton R S, Precup D, Singh S. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning[J]. _Artificial intelligence, 1999, 112(1-2): 181-211._
* [3] Van Hasselt H, Guez A, Silver D. Deep Reinforcement Learning with Double Q-Learning[C]. _AAAI. 2016: 2094-2100._
* [4] Del Moral P. Non-linear filtering: interacting particle resolution[J]. _Markov processes and related fields, 1996, 2(4): 555-581._
* [5] Bacon P L, Harb J, Precup D. The option-critic architecture[J]. _arXiv preprint arXiv:1609.05140, 2016._
* [6] Dayan P, Hinton G E. Feudal reinforcement learning[C]. _Advances in neural information processing systems. Morgan Kaufmann Publishers, 1993: 271-271._
* [7] Hasselt H V. Double Q-learning[C]. _Advances in Neural Information Processing Systems. 2010: 2613-2621._
* [8] Microsoft Co. Task and example code for the Malmo Collaborative AI Challenge, [_Github code_](https://github.com/Microsoft/malmo-challenge)

