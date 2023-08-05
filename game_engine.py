import numpy as np

# -1 -> blocks
# 1 -> group A
# 2 -> group B
BLOCK_VALUE = -1
GROUP_A_VALUE = 1
GROUP_B_VALUE = 2

# 0 -> left
# 1 -> right
# 2 -> up
# 3 -> down

class GroupMember:
    
    def __init__(self, group, cord, index, reward) -> None:
        self.group = group
        self.cord = cord
        self.index = index
        self.reward = reward
        

class Game:
    
    def __init__(self, size, state_fn, game_colddown=200) -> None:
        self.size = size
        self.state_fn = state_fn
        self.game_colddown = game_colddown
        
        self.members = [] # 0,1: A | 2: B
        
        for i in range(3):
            group = 'A' if i < 2 else 'B'
            self.members.append(GroupMember(group, (0, 0), i, 0))
        
        self.n_iterations = 0
        self.done = False
        
        self.__reset_grid()
        
    def __reset_grid(self):
        
        self.grid = np.zeros((3*self.size, 3*self.size), dtype=np.int32)
        
        self.grid[self.size:self.size*2, self.size:self.size*2] = BLOCK_VALUE
        
        self.grid = np.pad(self.grid, 1, constant_values=BLOCK_VALUE)
    
    def __reset_members_cords_and_rewards(self):
        for i in range(len(self.members)):
            if self.members[i].group == 'A':
                x_cord = np.random.randint(1, self.size)
                y_cord = np.random.randint(1, self.size*3)
            elif self.members[i].group == 'B':
                x_cord = np.random.randint(self.size*2+1, self.size*3) # +1 because of padding
                y_cord = np.random.randint(1, self.size*3)
            
            self.members[i].cord = (x_cord, y_cord)
            self.members[i].reward = 0
    
    def __collision_with_enemy_members(self): # group B dies
        if any(
            [
                (mem.cord == self.members[-1].cord or self.__distance(mem.cord, self.members[-1].cord) == 1) for mem in self.members[:-1]
            ]
        ):
            return True
        return False
    
    def __distance(self, cord1, cord2):
        return np.abs(cord1[0] - cord2[0]) + np.abs(cord1[1] - cord2[1])

    def __timeout(self): # group A dies
        if self.n_iterations > self.game_colddown:
            return True
        return False

    def __get_state(self):
        return self.state_fn(
            members=self.members,
            time=self.n_iterations/self.game_colddown,
            size=self.size
            ) # TODO
    
    def reset(self):
        self.__reset_grid()
        self.__reset_members_cords_and_rewards()
        self.n_iterations = 0
        self.done = False
        
        return self.__get_state()
    
    def __move(self, obj, move):
        # 1. object doesn't go into a friend member
        # 2. object doesn't go into wall
        
        if move == 0 or move == 1:
            step = -1 if move == 0 else 1
            
            # print([(obj.cord[1] + step == mobj.cord[1] and obj.cord[0] == mobj.cord[0]) for mobj in self.members[:-1]])
            if not any([(obj.cord[1] + step == mobj.cord[1] and obj.cord[0] == mobj.cord[0] and obj.group == mobj.group) for mobj in self.members])\
                and self.grid[obj.cord[0], obj.cord[1]+step] != BLOCK_VALUE:
                obj.cord = (obj.cord[0], obj.cord[1]+step)
            
        elif move == 2 or move == 3:
            step = -1 if move == 2 else 1
            
            if not any([(obj.cord[0] + step == mobj.cord[0] and obj.cord[1] == mobj.cord[1] and obj.group == mobj.group) for mobj in self.members])\
                and self.grid[obj.cord[0]+step, obj.cord[1]] != BLOCK_VALUE:
                obj.cord = (obj.cord[0]+step, obj.cord[1])
        
    
    def step(self, moves):
        # moves is a list moves for each member based on index
        # print(self.grid)
        # check game endings 1
        if self.__collision_with_enemy_members():
            self.done = True
            for i in range(3):
                self.members[i].reward = 10 if i != 2 else -10
        
        # check game endings 2
        if self.__timeout():
            self.done = True
            for i in range(3):
                self.members[i].reward = 10 if i == 2 else -10
        
        # make the move for all objects
        for i in range(3):
            self.__move(self.members[i], moves[i])
        
        # change grid
        self.__reset_grid()
        for i in range(3):
            self.grid[self.members[i].cord] = GROUP_A_VALUE if self.members[i].group == 'A' else GROUP_B_VALUE
        
        # change some variables
        
        self.n_iterations += 1
        
        return self.__get_state(), tuple([obj.reward for obj in self.members]), self.done