import gym
import numpy as np
from collections import Counter

env = gym.make('MountainCar-v0')
scores = []
training_data = []
accepted_scores = []
position_requirement = -0.3
env.reset()
for _ in range(1000):
    previous_observation = []
    game_memory = []
    max_position = -1
    for t in range(300):
        action = np.random.randint(3)  # env.action_space.sample()  #
        observation, reward, done, info = env.step(action)
        if len(previous_observation) > 0:
            game_memory.append(observation)
        previous_observation = observation
        if done:
            break
        if observation[0] > max_position:
            max_position = observation[0]

    if max_position >= position_requirement:
        accepted_scores.append(max_position)
        for data in game_memory:
            training_data.append(data)
    env.reset()

env.close()
print(np.array(training_data))
# print('Average accepted score:', mean(accepted_scores))
# print('Median score for accepted scores:', median(accepted_scores))
