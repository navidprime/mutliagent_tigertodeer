import numpy as np
 
distance_norm = 17 # for l = 4

def direction_between_two_cord(cord1, cord2):
    # left, right, up, down
    
    first_vs = np.zeros((4,), dtype=np.uint8)
    second_vs = np.zeros((4,), dtype=np.uint8)
    
    if cord1[0] < cord2[0]:
        first_vs[3] = 1
    elif cord1[0] > cord2[0]:
        first_vs[2] = 1
    if cord1[1] < cord2[1]:
        first_vs[1] = 1
    elif cord1[1] > cord2[1]:
        first_vs[0] = 1
    
    # to reverse: 
    # 1. left becamse right 
    # 2. up becames down and reverse
    
    if first_vs[0] == 1:
        second_vs[1] = 1
    elif first_vs[1] == 1:
        second_vs[0] = 1
    
    if first_vs[2] == 1:
        second_vs[3] = 1
    elif first_vs[3] == 1:
        second_vs[2] = 1
    
    return first_vs, second_vs

def direction_one_cord(cord1, cord2):
    first_vs = np.zeros((4,), dtype=np.uint8)
    
    if cord1[0] < cord2[0]:
        first_vs[3] = 1
    elif cord1[0] > cord2[0]:
        first_vs[2] = 1
    if cord1[1] < cord2[1]:
        first_vs[1] = 1
    elif cord1[1] > cord2[1]:
        first_vs[0] = 1

    return first_vs

def compute_blocks(cord, length):
    cord = cord[0]-1, cord[1]-1

    corners = np.zeros((4,), dtype=np.uint8)
    
    if cord[1] == 0:
        corners[0] = 1
    elif cord[1] == length*3-1:
        corners[1] = 1
    
    if cord[0] == 0:
        corners[2] = 1
    elif cord[0] == length*3-1:
        corners[3] = 1

    if not np.any(corners.astype(bool)): # obj is not near corners. so check the center block (this requiers some calculations)
        if cord[0] == length -1:
            if length-1 < cord[1] < length*2:
                corners[3] = 1
        elif cord[0] == length*2:
            if length-1 < cord[1] < length*2:
                corners[2] = 1
        elif cord[1] == length-1:
            if length-1 <= cord[0] <= length*2:
                corners[1] = 1 # TODO
        elif cord[1] == length*2:
            if length-1 <= cord[0] <= length*2:
                corners[0] = 1
        
    return corners
    
def distance(cord1, cord2):
    return (np.abs(cord1[0] - cord2[0]) + np.abs(cord1[1] - cord2[1])) / distance_norm

def get_state(**info): # info -> members, n_iterations/game_colddown
    state_A1 = []
    state_A2 = [] # [4](dir) + [1](dis) + [1](time) + [4](blocks) + [4](direction) + [1](distance)
    state_B1 = [] # [4](direction) + [4](direction) + [2](distance) + [4](blocks)
    
    # A: info about each other
    # - direction
    d1, d2 = direction_between_two_cord(info['members'][0].cord, info['members'][1].cord)
    
    state_A1.append(d1)
    state_A2.append(d2)
    # - distance
    d = distance(info['members'][0].cord, info['members'][1].cord)
    state_A1.append([d])
    state_A2.append([d])
    
    # A: remaining time
    state_A1.append([info['time']])
    state_A2.append([info['time']])
    
    # A: blocks
    
    state_A1.append(compute_blocks(info['members'][0].cord, info['size']))
    state_A2.append(compute_blocks(info['members'][1].cord, info['size']))
    
    # B <-> A: info about each group
    # - direction
    d1,r1 = direction_between_two_cord(info['members'][0].cord, info['members'][2].cord)
    d2,r2 = direction_between_two_cord(info['members'][1].cord, info['members'][2].cord)
    
    state_A1.append(d1)
    state_A2.append(d2)
    
    state_B1.append(r1)
    state_B1.append(r2)
    
    # - distance
    r1 = distance(info['members'][0].cord, info['members'][2].cord)
    r2 = distance(info['members'][1].cord, info['members'][2].cord)
    
    state_A1.append([r1])
    state_A2.append([r2])
    
    state_B1.append([r1])
    state_B1.append([r2])
    
    # B : blocks
    
    state_B1.append(compute_blocks(info['members'][2].cord, info['size']))
    
    return np.concatenate(state_A1), np.concatenate(state_A2), np.concatenate(state_B1)
    