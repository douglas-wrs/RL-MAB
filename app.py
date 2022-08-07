import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import plotly.express as px
import time
import pandas as pd

class Bandit:
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time_step = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
    
    def reset(self):
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time_step = 0
    
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time_step + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time_step += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time_step

        if self.sample_averages:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward

def simulate(runs, time_step, bandits):
    rewards = np.zeros((len(bandits), runs, time_step))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time_step):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards

def simulate(runs, time_step, bandits):
    rewards = np.zeros((len(bandits), runs, time_step))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in  trange(runs):
            bandit.reset()
            for t in range(time_step):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards

st.markdown('# 10-Armed Testbed')

st.markdown('#### Experiment Settings')
coluna = st.columns(4)

arms = coluna[0].number_input('Arms', value=10)
time_step = coluna[1].number_input('Steps', value=1000)
runs = coluna[2].number_input('Runs', value=10)
run_experiment = st.button('Run')

dataset = np.random.randn(200, int(arms)) + np.random.randn(int(arms))

fig = plt.figure(figsize=(12, 6))
fig = px.violin(dataset,
                box=True,
                labels={
                    "value": "Reward Distribution",
                    "variable": "Action"
                })
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
st.write(fig)

if run_experiment:
    with st.spinner('\u03B5-Greedy Experiment'):
        st.markdown('#### \u03B5-Greedy')
        epsilons = [0, 0.1, 0.01]
        bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]

        bandit_names = {x: f'\u03B5 = {epsilons[x]}' for x in range(len(bandits))}

        mean_best_action_counts, mean_rewards = simulate(runs, time_step, bandits)

        df_reward = pd.DataFrame(mean_rewards).transpose()
        df_action = pd.DataFrame(mean_best_action_counts).transpose()

        df_reward.columns = [ bandit_names[x] for x in df_reward.columns ]
        df_action.columns = [ bandit_names[x] for x in df_action.columns ]

        fig_reward = px.line(df_reward, labels={'index': 'Steps', 'value': 'Average Reward', 'variable': 'Bandits'})
        fig_action = px.line(df_action, labels={'index': 'Steps', 'value': 'Optimal Action', 'variable': 'Bandits'})
        st.write(fig_reward)
        st.write(fig_action)

    with st.spinner('Optimistic \u03B5-Greedy Experiment'):
        st.markdown('#### Optimistic \u03B5-Greedy')
        optimitict_values = [[0, 5], [0.1, 0]]
        bandits = []
        bandits.append(Bandit(epsilon=0, initial=5, step_size=0.1))
        bandits.append(Bandit(epsilon=0.1, initial=0, step_size=0.1))

        bandit_names = {x: f'\u03B5 = {optimitict_values[x][0]}, q = {optimitict_values[x][1]}' for x in range(len(bandits))}

        mean_best_action_counts, mean_rewards = simulate(runs, time_step, bandits)

        df_action = pd.DataFrame(mean_best_action_counts).transpose()

        df_action.columns = [ bandit_names[x] for x in df_action.columns ]

        fig_action = px.line(df_action, labels={'index': 'Steps', 'value': 'Optimal Action', 'variable': 'Bandits'})

        st.write(fig_action)

    with st.spinner('UCB Experiment'):
        st.markdown('#### Upper Confidence Bound')
        paramenters_bandits = [['-', 2], [0.1, '-']]
        bandits = []
        bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
        bandits.append(Bandit(epsilon=0.1, sample_averages=True))
        
        _, average_rewards = simulate(runs, time_step, bandits)

        def name_bandit(x):
            if x[0] == '-':
                return f'UCB c = {x[1]}'
            elif x[1] == '-':
                return f'\u03B5-Greedy \u03B5 = {x[0]}'

        bandit_names = {x: name_bandit(paramenters_bandits[x]) for x in range(len(bandits))}

        mean_best_action_counts, mean_rewards = simulate(runs, time_step, bandits)

        df_reward = pd.DataFrame(mean_rewards).transpose()

        df_reward.columns = [ bandit_names[x] for x in df_reward.columns ]

        fig_reward = px.line(df_reward, labels={'index': 'Steps', 'value': 'Average Reward', 'variable': 'Bandits'})

        st.write(fig_reward)

    with st.spinner('Gradient Experiment'):
        st.markdown('#### Gradient Bandit')
        bandits = []
        bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))
        bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4))
        bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4))
        bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4))

        labels = ['\u03B1 = 0.1, with baseline',
                '\u03B1 = 0.1, without baseline',
                '\u03B1 = 0.4, with baseline',
                '\u03B1 = 0.4, without baseline']

        bandit_names = {x: labels[x] for x in range(len(bandits))}

        mean_best_action_counts, mean_rewards = simulate(runs, time_step, bandits)

        df_action = pd.DataFrame(mean_best_action_counts).transpose()

        df_action.columns = [ bandit_names[x] for x in df_action.columns ]

        fig_action = px.line(df_action, labels={'index': 'Steps', 'value': 'Optimal Action', 'variable': 'Bandits'})
        st.write(fig_action)

    with st.spinner('Bandits Comparision Experiment'):
        st.markdown('#### Bandits Comparision')
        labels = ['\u03B5-greedy', 'gradient bandit',
                    'UCB', 'optimistic initialization']
                    
        generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                    lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                    lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                    lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]

        parameters = [np.arange(-7, -1, dtype=np.float),
                    np.arange(-5, 2, dtype=np.float),
                    np.arange(-4, 3, dtype=np.float),
                    np.arange(-2, 3, dtype=np.float)]

        bandits = []
        bandits_index = []
        parameters_values = {}
        count_bandit = 0
        count_parameter = 0
        for generator, parameter in zip(generators, parameters):
            for param in parameter:
                bandits.append(generator(pow(2, param)))
                bandits_index.append(count_bandit)
                parameters_values[count_parameter] = param
                count_parameter += 1
            count_bandit += 1

        bandit_names = {0 : '\u03B5-Greedy',
                        1 : 'Gradient Bandit',
                        2 : 'UCB',
                        3 : 'Optimistic Initialization'}

        _, average_rewards = simulate(runs, time_step, bandits)
        rewards = np.mean(average_rewards, axis=1)
        df_reward = pd.DataFrame(rewards).reset_index()
        df_reward = df_reward.rename(columns={'index': 'bandit', 0: 'reward'})

        bandits_dict = dict(zip(range(len(bandits_index)), bandits_index))
        df_reward['bandit'] = df_reward.bandit.apply(lambda x: bandits_dict[x])
        df_reward['bandit'] = df_reward.bandit.apply(lambda x: bandit_names[x])
        df_reward.reset_index(inplace=True)
        df_reward["parameter"] = df_reward['index'].apply(lambda x: parameters_values[x])

        fig_reward = px.line(df_reward.reset_index(), x='parameter', y='reward',
        labels={'parameter': 'Parameters', 'bandit': 'Bandits', 'reward': 'Average Reward'}, color='bandit')
        st.write(fig_reward)
else:
    st.stop()