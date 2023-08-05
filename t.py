import numpy as np
l = 2
a = np.zeros((l*3, l*3), dtype=np.int32)
a[l:l*2, l:l*2] = -1
# a = np.pad(a, 1, constant_values=-1)

# print(a)

# print([0,1,2][:-1])

# print(np.argwhere(np.array([[0, 1, 0, 0],
#                             [0, 0, 1, 1]])))

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
    # 2. up becames down 
    
    if first_vs[0] == 1:
        second_vs[1] = 1
    elif first_vs[1] == 1:
        second_vs[0] = 1
    
    if first_vs[2] == 1:
        second_vs[3] = 1
    elif first_vs[3] == 1:
        second_vs[2] = 1
    
    return first_vs, second_vs
def distance(cord1, cord2):
    return np.abs(cord1[0] - cord2[0]) + np.abs(cord1[1] - cord2[1])
# print(distance((5, 5 ), (4, 4)))

def compute_corners(cord, length):
    corners = np.zeros((4,))
    
    if cord[1] == 0:
        corners[0] = 1
    elif cord[1] == length-1:
        corners[1] = 1
    
    if cord[0] == 0:
        corners[2] = 1
    elif cord[0] == length-1:
        corners[3] = 1
    
    return corners

# print(compute_corners((0,4), 5))

# print(a)
# print(a[l-1:l*2+2-1, l-1:l*2+2-1])

def compute_blocks(cord, length): # l
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
        print('here')
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

print(np.pad(a, 1 , constant_values=-1))
print(np.pad(a, 1 , constant_values=-1).shape)
print(compute_blocks((3,2), 2))


i = 0.01
n= 0
while n < 1500:
    n += 1
    i = i * np.exp(-0.003)
    
    print(n, i)