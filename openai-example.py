import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

# random_games()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            env.render()
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    print('Average accepted scores:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print('Counter accepted score:', Counter(accepted_scores))

    return training_data

def neural_network(input_size):
    net = input_data(shape=[None, input_size, 1], name='input')
    # Hidden layer 1
    net = fully_connected(net, 128, activation='relu')
    net = dropout(net, 0.8)
    # Hidden layer 2
    net = fully_connected(net, 256, activation='relu')
    net = dropout(net, 0.8)
    # Hidden layer 3
    net = fully_connected(net, 512, activation='relu')
    net = dropout(net, 0.8)
    # Hidden layer 4
    net = fully_connected(net, 256, activation='relu')
    net = dropout(net, 0.8)
    # hidden layer 5
    net = fully_connected(net, 128, activation='relu')
    net = dropout(net, 0.8)
    # Output layer
    net = fully_connected(net, 2, activation='softmax')
    net = regression(net, optimizer='adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(net, tensorboard_dir='log')
    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network(input_size=len(X[0]))

    model.fit({'input':X}, {'target':y}, n_epoch=3, snapshot_step=500,
              show_metric=True, run_id='openaigame')
    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)
        new_obs, reward, done, info = env.step(action)
        prev_obs = new_obs
        game_memory.append([new_obs, action])
        score += reward
        if done:
            break
    scores.append(score)
print('Average Score:', sum(scores)/len(scores))
print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices),
                                          choices.count(0)/len(choices)))
