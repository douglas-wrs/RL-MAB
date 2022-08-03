import streamlit as st
import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

np.random.seed(30)
env = gym.make("BanditTenArmedGaussian-v0", bandits = 10) 
observation = env.reset()

N_list = []
e_greedy = (st.number_input(label='Epsilon-greedy', min_value=0, max_value=100, step=1, value=10))/100
space_options = [1,10]
options = range(*space_options)
Q_dict = {a:0 for a in options}
step_list = []
rewards_list = []

last_rows = np.array([0.0])
chart = st.line_chart(last_rows)

progress_bar = st.progress(0)
# status_text = st.empty()

t_max = 500
for i_episode in range(1,t_max):
    if random.random() < 1-e_greedy:
        # print('exploit')
        action = int(np.argmax(np.array(Q_dict.values()))+1)
    else:
        # print('explore')
        action = random.randint(min(options), max(options))
        
    if i_episode == 1:
        action = random.randint(min(options), max(options))
    
    observation, reward, done, info = env.step(action)
    N_list.append(action)
    N_a = N_list.count(action)
    N_a = 0.00000000000001 if N_a == 0 else N_a
    Q_dict[action] = Q_dict[action] + (1/N_a)*(reward-Q_dict[action])
    
    step_list.append(i_episode)
    rewards_list.append(reward)
    new_row = np.array([(sum(rewards_list)/i_episode)])
    last_rows = np.append(last_rows[1:], new_row)
    chart.add_rows(last_rows)
    
    etapa = i_episode/t_max
    # status_text.text("%i%% Complete" % etapa)
    progress_bar.progress(etapa)
    time.sleep(0.001)
env.close()

progress_bar.empty()
st.button("Re-run")