from game import GameWithGraphics
from state_setting import get_state
from multi_agent import Agents
from pynput import keyboard
import numpy as np
import time

def on_press(k):
    global B_action
    try:
        # 2 -> left, 3 -> right, 0 -> down, 1 -> up
        if k.char == 'w': 
            B_action = [1] 
        elif k.char == 's':
            B_action = [0]
        elif k.char == 'a':
            B_action = [2]
        elif k.char == 'd':
            B_action = [3]
    except AttributeError:
        pass

fps = 2
length = 4

def main():
    global B_action
    
    game = GameWithGraphics(length, get_state, 70)
    
    state = game.reset()
    B_action = [0]
    
    agents = Agents([len(s) for s in state],
                    [4]*3,
                    2048,
                    lambda n, lr: lr if n < 180 else lr*np.exp(-.018),
                    .009,
                    epsilon_length=200)
    
    agents.load_models(['./saved_models/0', './saved_models/1',
                        './saved_models/2'])
    agents.set_test_mode(True)
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    for i in range(80_000):
        state = list(state)
        state[0][5] = .99 # they realy did take things seriously
        state[1][5] = .99
        actions = agents(state)
        actions = [a.numpy().item() for a in actions]
        next_state, rewards, done = game.step(
            np.concatenate([actions[:-1], B_action])
        )
        
        state = next_state
        
        if done:
            state = game.reset()
        
        time.sleep(1/fps)
        
if __name__ == '__main__':
    main()