import turtle
import numpy as np
from game_engine import Game, GroupMember

COLORS = [
          '#f8fafc', # white
          '#dc2626', # red
          '#2563eb',# blue
          '#020617', ]# black

BLOCK_SIZE = 20
FONT_SIZE = 20
W,H = 600, 600

class GameWithGraphics(Game):
    
    def __init__(self, size, state_fn, game_colddown=200) -> None:
        super().__init__(size, state_fn, game_colddown)
        
        self.member_turtles =[
            self.__make_turtle_obj(obj) for obj in self.members
        ]
        
        self.is_blocks_drawed = False # we will show blocks only once
        
        self.offset = size*BLOCK_SIZE*3/2 # used to center the board
        
        self.wn = turtle.Screen()
        self.wn.setup(W, H)
        self.wn.bgcolor(COLORS[-1])
        self.wn.tracer(0)
    
    def __make_turtle_obj(self, mem_obj): # used for only members
        
        color = 1 if mem_obj.group == 'A' else 2
        
        t = turtle.Turtle('square')
        t.index = mem_obj.index
        t.speed(0)
        t.color(COLORS[color])
        t.penup()
        
        return t

    def __draw_blocks(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i, j] == -1:
                    t = turtle.Turtle('square')
                    t.speed(0)
                    t.color(COLORS[0])
                    t.penup()
                    t.goto(i*BLOCK_SIZE - self.offset, j*BLOCK_SIZE - self.offset)
    
    def __move_turtle_object(self, obj):
        for i in range(3):
            obj_cord = self.members[obj.index].cord
            
            obj.goto(obj_cord[0]*BLOCK_SIZE - self.offset, obj_cord[1]*BLOCK_SIZE - self.offset)
    
    def reset(self):
        if not self.is_blocks_drawed:
            self.__draw_blocks()
            self.is_blocks_drawed = True

        for obj in self.member_turtles:
            self.__move_turtle_object(obj)
            
        return super().reset()
    
    def step(self, moves):
        for obj in self.member_turtles:
            self.__move_turtle_object(obj)
        
        self.wn.update()
        
        return super().step(moves)