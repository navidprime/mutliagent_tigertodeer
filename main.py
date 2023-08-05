from game import GameWithGraphics
from state_setting import get_state
from multi_agent import Agents

import numpy as np

length = 4

def main():
    
    matches_won = list()
    
    game = GameWithGraphics(length, get_state, 40)
    
    state = game.reset()
    
    agents = Agents([len(s) for s in state],
                    [4]*3,
                    4096,
                    lambda n, lr: lr if n < 300 else lr*np.exp(-0.001),
                    .002,
                    epsilon_length=5)
    
    agents.load_models(['./models_in_90timeout/0', './models_in_90timeout/1',
                        './models_in_90timeout/2']) # i'm retraining the previous saved models
    
    for i in range(40_000):
        actions = agents(state)
        actions = [a.numpy().item() for a in actions]
        next_state, rewards, done = game.step(
            actions
        )
        
        agents.train_and_remember(state, actions, rewards, next_state, done, False)
        
        state = next_state
        
        if done:
            print('-'*100)
            print('n iters: ', i)
            print('n games: ', agents.agents[0].n_games.numpy().item())
            print('last 100 matchs group A won (Percentage): ', np.sum(matches_won), '/', len(matches_won))
            
            if len(matches_won) > 100:
                matches_won = list()
            
            if rewards[-1] < 0:
                matches_won.append(1) # A win
            elif rewards[-1] > 0:
                matches_won.append(0) # B win
            
            agents.when_episode_done()
            agents.train_long()
            
            state = game.reset()
        
    agents.save_models()

if __name__ == '__main__':
    main()