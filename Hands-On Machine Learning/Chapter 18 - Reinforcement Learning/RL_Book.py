# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 07:34:09 2021

@author: Z52XXR7

Chapter 18 - Reinforcement Learning

"""

import gym
import numpy as np

env = gym.make('CartPole-v1')
obs = env.reset()
# env.render()

img = env.render(mode="rgb_array") # Transform the environment into a image

img.shape

env.action_space # Shows the possible actions

action = 1 # accelerate right

obs, reward, done, info = env.step(action)

#Run the environment and computes the rewards

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
    
    
np.mean(totals), np.std(totals), np.min(totals), np.max(totals)
    

#Neural Network Policies

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

n_inputs = 4 # env.observation

# model = keras.models.Sequential[(
#     keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
#     keras.layers.Dense(1, activation="sigmoid"),
#     )]

model = Sequential()
model.add(Dense(5, activation='elu',input_shape=[n_inputs]))
model.add(Dense(1, activation="sigmoid"))

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1,1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0,0].numpy()))
    return obs, reward, done, grads
        

# Play multiple episodes with theirs gradients and rewards
def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) -2, -1, -1):
        discounted[step] += discounted[step + 1]*discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]

discount_rewards([10, 0, -50], discount_factor=0.8)
discount_and_normalize_rewards([[10,0, -50],[10,20]], discount_factor=0.8) 

n_interations = 15
n_episodes_per_update = 5
n_max_steps = 20
discount_factor = 0.95

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.binary_crossentropy

# Build and run the training loop

for interation in range(n_interations):
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn)
    all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                        discount_factor)
    
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward*all_grads[episode_index][step][var_index]
            for episode_index, final_reward in enumerate(all_final_rewards)
                for step, final_reward in enumerate(final_reward)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
            