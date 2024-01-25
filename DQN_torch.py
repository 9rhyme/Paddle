import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from paddle_pygame import Paddle
from datetime import datetime

env = Paddle()
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



class Agent:

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        self.memory = deque(maxlen=100000)
        self.model = DQN(state_space, action_space).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss().to(device)
        if not env.train_new:
            try:
                self.model.load_state_dict(torch.load(env.model_to_use))
                self.model.eval()
                print("Loaded model")
            except FileNotFoundError:
                print("No saved model found. Creating a new model.")

    def save_model(self , current_datetime):

        model_filename = f'saved_model_{current_datetime}.pth'
        torch.save(self.model.state_dict(), model_filename)
        print(f"Model saved to '{model_filename}'")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = torch.FloatTensor(state).to(device)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = torch.from_numpy(states.squeeze()).float().to(device)
        next_states = torch.from_numpy(next_states.squeeze()).float().to(device)

        self.model.eval()
        next_state_values = self.model(next_states).detach().numpy()
        self.model.train()

        targets = rewards + self.gamma * (np.amax(next_state_values, axis=1)) * (1 - dones)
        targets_full = self.model(states)
        targets_pred = targets_full.clone()
        targets_full[np.arange(self.batch_size), actions] = torch.from_numpy(targets).float()
        # value_pred = self.model(states)
        loss = self.MSE_loss(targets_pred, targets_full)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):
    loss = []
    accuracy_list = []  # List to store accuracy for each episode
    action_space = 3
    state_space = 5

    agent = Agent(state_space, action_space)

    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, [1, state_space])
        score = 0
        done = False  # Flag to track if the episode should end

        while not done:
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_space])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            score += reward
        accuracy = env.hit / (env.hit + env.miss) * 100 if (env.hit + env.miss) > 0 else 0
        accuracy_list.append(accuracy)
        print("episode: {}/{}, score: {}, accuracy: {:.2f}%".format(e, episode, score, accuracy))
        loss.append(score)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_filename = f'graph_{current_datetime}.png'
    agent.save_model(current_datetime)
    plt.plot([i for i in range(ep)], accuracy_list)
    plt.xlabel('episodes')
    plt.ylabel('accuracy %')
    plt.title('Training Accuracy Over Episodes')
    plt.savefig(graph_filename)
    plt.show()
    return accuracy_list


if __name__ == '__main__':
    ep = 1000
    accuracy = train_dqn(ep)

