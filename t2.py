from game import Game
from state_setting import get_state

import numpy as np
import time

fps = 2
length = 2

def main():
    game = Game(length, get_state, 50)
    
    state = game.reset()
    
    for i in range(150_000):
        actions = np.random.randint(4, size=(4,)).tolist()
        next_state, rewards, done = game.step(
            actions
        )
        
        if rewards[0] > 0:
            print(rewards)
            print(done)
        elif rewards[-1] > 0:
            print(rewards)
            print(done)
        
        state = next_state
        
        
        if done:
            state = game.reset()
        
        time.sleep(1/fps)

if __name__ == '__main__':
    main()
    # i = .01
    # n = 0
    # while i > .00001:
    #     i = i *np.exp(-.019)
    #     n += 1
    #     print(i, n)