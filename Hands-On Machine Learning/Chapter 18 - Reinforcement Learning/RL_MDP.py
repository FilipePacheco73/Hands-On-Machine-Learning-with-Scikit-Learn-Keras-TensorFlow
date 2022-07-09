# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 07:34:09 2021

@author: Z52XXR7

Chapter 18 - Reinforcement Learning - MDP

"""

import numpy as np

transition_probabilities = [ #shape=[s,a,s']
                            [[0.7,0.3,0.0],[1,0,0],[.8,.2,0]],
                            [[0,1,0], None, [0, 0, 1]],
                            [None, [0.8,0.1,0.1], None]]

rewards = [ # shape=[s,a,s']
           [[+10,0,0],[0,0,0],[0,0,0]],
           [[0,0,0],[0,0,0],[0,0,-50]],
           [[0,0,0],[40,0,0],[0,0,0]]]

possible_actions = [[0,1,2],[0,2],[1]]

#Set up initial Q-values, Q-values impossible set as -infinity

Q_values = np.full((3,3), -np.inf) # -np.inf for impossible actions

for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0
    
gamma = 0.95 # the discount factor

for interation in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s,a] = np.sum([
                transition_probabilities[s][a][sp]
                *(rewards[s][a][sp] + gamma*np.max(Q_prev[sp]))
            for sp in range(3)])
            
np.argmax(Q_values, axis=1) # optimial action for each state

# Q-Learning
def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0,1,2], p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward

def exploration_policy(state):
    return np.random.choice(possible_actions[state])

alpha0 = .05
decay = .005
gamma = .9
state = 0

for interation in range(10000):
    action = exploration_policy(state)
    next_state, reward = step(state,action)
    next_value = np.max(Q_values[next_state])
    alpha = alpha0/(1+interation*decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha*(reward + gamma*next_value)
    state = next_state

