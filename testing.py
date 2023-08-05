from game import GameWithGraphics
from state_setting import get_state
from multi_agent import Agents
import numpy as np
import time

fps = 2
length = 4

def main():
    game = GameWithGraphics(length, get_state, 70)
    
    state = game.reset()
    
    agents = Agents([len(s) for s in state],
                    [4]*3,
                    2048,
                    lambda n, lr: lr if n < 180 else lr*np.exp(-.018),
                    .009,
                    epsilon_length=200)
    
    agents.load_models(['./saved_models/0', './saved_models/1',
                        './saved_models/2'])
    
    agents.set_test_mode(True)
    
    for i in range(80_000):
        actions = agents(state)
        actions = [a.numpy().item() for a in actions]
        next_state, rewards, done = game.step(
            actions
        )
        
        state = next_state
        
        if done:
            state = game.reset()
        
        time.sleep(1/fps)
        
if __name__ == '__main__':
    main()