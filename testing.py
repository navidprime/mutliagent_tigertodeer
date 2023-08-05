from game import GameWithGraphics
from state_setting import get_state
from multi_agent import Agents

import numpy as np
import time

fps = 1
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
    
    agents.load_models(['./0','./1','./2'])
    agents.set_test_mode(True)
    
    for i in range(80_000):
        # print(state[0][-5:])
        # print(state[1][-5:])
        print(state[0])
        # print(state[2])
        actions = agents(state)
        actions = [a.numpy().item() for a in actions]
        # print(actions)
        next_state, rewards, done = game.step(
            actions
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