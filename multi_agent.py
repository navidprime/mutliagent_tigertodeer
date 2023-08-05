from typing import Any
from agent import Agent
import tensorflow as tf

class Agents:
    
    def __init__(self, inputs, outputs, batch_size, lr_schduler,lr=0.005, gamma=.99, epsilon_length=150, n=3) -> None:
        self.n = n
        
        self.agents = [
            Agent(outputs[i], inputs[i], batch_size, lr_schduler, lr, gamma, epsilon_length)
            for i in range(len(inputs))
        ]
    def train_and_remember(self, states, actions, rewards, next_states, done, truncate):
        for i in range(len(self.agents)):
            self.agents[i].remember_and_train_short(states[i], actions[i], rewards[i], next_states[i], done, truncate)
    
    def train_long(self):
        for i in range(len(self.agents)):
            self.agents[i].train_long()
    
    def when_episode_done(self):
        for i in range(len(self.agents)):
            self.agents[i].when_episode_done()
    
    def __call__(self, state):
        actions = [
            self.agents[i](state[i])
            for i in range(len(self.agents))
        ]
        
        return actions

    def save_models(self, path='./'):
        for i in range(len(self.agents)):
            self.agents[i].model.save(path+str(i), save_format='tf')
    
    def load_models(self, pathes):
        for i in range(len(self.agents)):
            self.agents[i].model = tf.keras.models.load_model(pathes[i])
    
    def set_test_mode(self, mode):
        for i in range(len(self.agents)):
            self.agents[i].test_mode.assign(mode)