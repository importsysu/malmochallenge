#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 00:01:53 2017

@author: alex
"""
#==============================================================================
# BadAgent represents a class of agents who favor going to the lapis block rather than chasing the pig
#==============================================================================
import sys
sys.path.append('../')
from collections import deque, namedtuple
from common import ENV_ACTIONS
from heapq import heapify, heappop, heappush

class BadAgent(object):
    def __init__(self, name):
        self.name = name
        self._previous_target_pos = None
        self._action_list = []
        self.ACTIONS = ENV_ACTIONS
        self.Neighbour = namedtuple('Neighbour', ['cost', 'x', 'z', 'direction', 'action'])
        
    def act(self, obs, reward, done, is_training = True):
        if done:
            self._action_list = []
            self._previous_target_pos = None
        #print state
        if obs is None:
            return 0
        entities = obs[1]
        state = obs[0]
        
        me = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k]
        me_details = [e for e in entities if e['name'] == self.name][0]
        yaw = int(me_details['yaw'])
        direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.
        if self.manhattan_dist(me[0], (1,4)) < self.manhattan_dist(me[0], (7,4)):
            goal = (1,4)
        else:
            goal = (7,4)
        target = [(goal[0], goal[1])]
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

    	# now prune:
        valid_neighbors = [n for n in neighbors if
                            n.x >= 0 and n.x < state_width and n.z >= 0 and n.z < state_height and state[
                                n.z, n.x] != 'sand']
        
    
        return valid_neighbors

    def heuristic(self, a, b, state=None):
        (x1, y1) = (a.x, a.z)
        (x2, y2) = (b.x, b.z)
        return abs(x1 - x2) + abs(y1 - y2)
    
    def matches(self, a, b):
        return a.x == b.x and a.z == b.z

    def manhattan_dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
