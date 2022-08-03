from ssl import ALERT_DESCRIPTION_ACCESS_DENIED
import streamlit as st
import gym_bandits
import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import math

st.markdown("# Multi-Armerd Bandits MAB")
env = gym.make("BanditTenArmedGaussian-v0")
env.reset()

coluna = st.columns(4)
coluna[0].markdown('### $\epsilon$-greedy')
epsilon = coluna[0].number_input('Epsilon', value=0.1)
initial_value = coluna[0].number_input('Initialization', value=0)
seed = coluna[0].number_input('Seed', value=64)
num_steps = coluna[0].number_input('Steps', value=1000)

def epsilon_greedy(epsilon):
    rand = np.random.random()
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    return action

count = np.empty(10)
count.fill(0)
sum_rewards = np.empty(10)
sum_rewards.fill(0)
mean_rewards = np.empty(10)
sum_rewards.fill(initial_value)
Q = np.empty(10)
Q.fill(5)
arm_history = []

reward_canvas = st.empty()
count_canvas = st.empty()
for i in range(int(num_steps)):
    arm = epsilon_greedy(epsilon)

    observation, reward, done, info = env.step(arm)

    count[arm] += 1
    sum_rewards[arm] += abs(reward)
    mean_rewards[arm] = abs(reward)

    Q[arm] = sum_rewards[arm]/count[arm]

    arm_history.append(mean_rewards.tolist())
    
    df_reward = pd.DataFrame(arm_history)
    df_reward_tmp = df_reward.reset_index().rename(columns={'index': 'step'})
    df_reward_step = pd.melt(df_reward_tmp, id_vars=['step'], value_vars=list(range(10)))
    df_reward_step.rename(columns={'variable': 'arm', 'value': 'reward'}, inplace=True)
    df_reward_mean = df_reward_step.groupby('step').agg(mean_reward=('reward', 'mean'))

    reward_canvas.line_chart(df_reward_mean)
    
    df_count = pd.DataFrame(count).rename(columns={0: 'count'})
    count_canvas.bar_chart(df_count)




