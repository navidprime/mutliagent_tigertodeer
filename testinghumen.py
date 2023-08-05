from game import GameWithGraphics
from state_setting import get_state
from multi_agent import Agents
from pynput import keyboard

def on_press(k):
    global B_action
    try:
        if k.char == 'w': # 2 -> left, 3 -> right, 0 -> down, 1 -> up
            B_action = [1] 
        elif k.char == 's':
            B_action = [0]
        elif k.char == 'a':
            B_action = [2]
        elif k.char == 'd':
            B_action = [3]
    except AttributeError:
        pass

import numpy as np
import time

fps = 2
length = 4

def main():
    global B_action
    game = GameWithGraphics(length, get_state, 70)
    
    state = game.reset()
    B_action = [0]
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    agents = Agents([len(s) for s in state],
                    [4]*3,
                    2048,
                    lambda n, lr: lr if n < 180 else lr*np.exp(-.018),
                    .009,
                    epsilon_length=200)
    
    agents.load_models(['./models_in40timeout/0', './models_in40timeout/1',
                        './models_in40timeout/2'])
    agents.set_test_mode(True)
    
    for i in range(80_000):
        # print(state[0][-5:])
        # print(state[1][-5:])
        # print(state[0])
        # print(state[2])
        actions = agents(state)
        actions = [a.numpy().item() for a in actions]
        # print(actions)
        next_state, rewards, done = game.step(
            np.concatenate([actions[:-1], B_action])
        )
        # agents.train_and_remember(state, actions, rewards, next_state, done, False)
        
        state = next_state
        
        if done:
            # print(i, agents.agents[0].n_games.numpy().item())
            # agents.when_episode_done()
            # agents.train_long()
            
            state = game.reset()
        
        time.sleep(1/fps)
        
    agents.save_models()

if __name__ == '__main__':
    main()