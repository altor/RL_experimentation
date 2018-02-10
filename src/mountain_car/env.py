import numpy as np
import cv2

import math
import gym

from decimal import *

from core.env import Environment, Markovian_State, State

def round(n, nb_decimals):
    return int(n * pow(10, nb_decimals))

class MC_State_gym_discrete(State):
    def __init__(self, gym_state, reward, is_terminal, grid_shape, nb_step, max_step):
        """
        Représentation discrète d'un état à partir des informations donné par gym
        """
        State.__init__(self, reward, [-1,0,1], False)
        
        self.velocity = gym_state[1]
        self.position = gym_state[0]
        if self.position >= 0.5:
            self.reward = 1
        else:
            self.reward = reward
        self.grid_shape = grid_shape
        self.is_terminal_bool = is_terminal
        self.nb_step = nb_step
        self.max_step = max_step
        
        self.id = (self.velocity, self.position)

    def is_terminal(self):
        # if self.nb_step >= self.max_step:
        #     return True
        if self.position >= 0.5:
            # print("toto")
            return True
            # print(str(self.nb_step))


        return False
        

    def to_space_grid_coord(self):
        x, y = self.grid_shape
        v, p = self.id

        v2 = int((v + 0.07) * x / 0.14)
        p2 = int((p + 1.2) * y / 1.8)
        return v2, p2
        # v, p = self.id
        # v2 = round(0.07, self.grid_shape) + v
        # p2 = round(1.2, self.grid_shape) + p
        # return v2, p2

class MC_gym_environment(Environment):
    """
    Environement du problème mountain car
    utilise l\'environemnet MountainCar-v0 de gym
    """

    def __init__(self, grid_shape, render_bool=False, max_step=200):
        Environment.__init__(self, init_state=None)
        self.gym_env = gym.make('MountainCar-v0')
        self.grid_shape = grid_shape
        self.nb_step = 0
        self.episode = 0
        self.render_bool = render_bool
        self.max_step = max_step
        
        self.re_init()

    def re_init(self):
        gym_state = self.gym_env.reset()
        self.nb_step = 0
        self.current_state = MC_State_gym_discrete(gym_state, -1,
                                                   False,
                                                   self.grid_shape, self.nb_step, self.max_step)
        self.gym_env.reset()
        self.episode += 1

    def render(self, txt1=None, txt2=None, txt3=None, waitkey=False):
        img = self.gym_env.render(mode='rgb_array')
        img = cv2.putText(img.copy(), str(self.episode), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
        if txt1 != None:
            img = cv2.putText(img.copy(), str(txt1), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
        if txt2 != None:
            img = cv2.putText(img.copy(), str(txt2), (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
        if txt3 != None:
            img = cv2.putText(img.copy(), str(txt3), (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))

        img = cv2.putText(img.copy(), 'v:' + str(self.current_state.velocity), (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
        img = cv2.putText(img.copy(), 'p:' + str(self.current_state.position), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
            
        cv2.imshow('toto', img)
        if waitkey:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
        
    def next_sa(self, action):
        if self.render_bool:
            self.render()
        self.nb_step += 1
        gym_state, r, is_terminal, _ = self.gym_env.step(action + 1)

        self.current_state = MC_State_gym_discrete(gym_state, r,
                                                   is_terminal,
                                                   self.grid_shape, self.nb_step, self.max_step)
        return self.current_state
    


class MC_State_naive_discrete(Markovian_State):
    def __init__(self, init_velocity, init_position, grid_shape):
        """
        représentation discrète d'un état pour le problème de la voiture dans la montagne.
        La vitesse et la position sont arrondies à un nombre de décimale données

        :param grid_shape: nombre de décimales auxquelles sont arondie 
                            les valeurs de l'état
        
        """
        Markovian_State.__init__(self, -1, -1)


        if(init_position < -1.2):
            print(init_position)
            raise ValueError
        
        self.velocity = init_velocity
        self.position = init_position
        self.grid_shape = grid_shape
        
        if self.position >= 0.5:
            self.reward = 1

        self.id = (round(self.velocity, grid_shape),
                   round(self.position, grid_shape))

        for action in [-1, 0, 1]:
            v, p = self.compute_next_state(action)
            self.add_transition((round(v, grid_shape),
                                 round(p, grid_shape)), action, 1)

    def next(self, action):
        v, p = self.compute_next_state(action)
        return MC_State_naive_discrete(v, p, self.grid_shape)
            
    def is_terminal(self):
        return self.position >= 0.6

    def compute_next_state(self, action):
        """
        :param action: -1|0|1
        :return: (new_velocity, new_position)
        """
        v = (self.velocity + (action * 0.001) +
             np.cos(np.radians((3 * self.position)) * (-0.0025)))
        return (v, self.position + v)
