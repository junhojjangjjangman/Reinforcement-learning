from collections import defaultdict
import pickle
import random
from IPython.display import clear_output
import numpy as np

import click
import pygame
import gym


ALPHA=0.4 #알파 값은 모델이 학습하는 속도를 나타냅니다.
GAMMA=0.9 #감마 값은 모델이 학습하는 방법을 결정하는 데 중요합니다. 감마가 너무 높으면 모델은 멀리서 크게 보고, 감마가 낮으면 너무 가깝게 자세히 봅니다.
EPSILON=0.1 #우리가 과거의 실패로부터 더 많은 것을 배울 수 있는 모델을 원할 때 엡실론 값을 높일 수 있습니다.
TIMES = 10000000

alpha = 0.1
gamma = 0.6
epsilon = 0.1

NUM_EPISODES = 100000

env = gym.make("Taxi-v3", render_mode="human")
obs = env.reset()
print(obs)
action = 5
env.step(action)
env.render()
# actions = env.action_space.n #taxi의 action개수
# print(actions)
# observations = env.observation_space.n #taxi의 state 개수
# print(observations)

#----------------------입력을 받아서 게임해보기 ---------------------------------
# done = False
# while not done:
#     env.render() # 이 줄에서 환경 렌더링 함수를 적용합니다.
#     i = int(input())
#     clear_output(wait=True)
#     obs,reward,complete,info,aa = env.step(i) # 여기에서 환경에 대한 단계를 실행
#     print('Observation = ', obs, '\nreward = ', reward, '\ndone = ', complete, '\ninformation = ', info)
#     done = complete
# env.close()
#----------------------입력을 받아서 게임해보기 ---------------------------------

q_table = np.zeros([env.observation_space.n, env.action_space.n])

all_epochs = []
all_penalties = []

for i in range(1, NUM_EPISODES + 1):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:  # 이전 지식을 사용하는 대신 새로운 행동을 탐구할 확률이 10%입니다.
            action = env.action_space.sample()  # 작업 공간 탐색
        else:
            action = np.argmax(q_table[state])  # 학습된 값 이용

        next_state, reward, done, info = env.step(action)  # 다음 단계를 수행합니다.

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

# 평가
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")