import numpy as np
import math
import gym

from decimal import *

from core.env import Environment, Markovian_State

def round(n, nb_decimals):
    return int(n * pow(10, nb_decimals))

class MC_State_naive_discrete(Markovian_State):
    def __init__(self, init_velocity, init_position, nb_decimals):
        """
        représentation discrète d'un état pour le problème de la voiture dans la montagne.
        La vitesse et la position sont arrondies à un nombre de décimale données

        :param nb_decimals: nombre de décimales auxquelles sont arondie 
                            les valeurs de l'état
        
        """
        Markovian_State.__init__(self, -1, -1)


        if(init_position < -1.2):
            print(init_position)
            raise ValueError
        
        self.velocity = init_velocity
        self.position = init_position
        self.nb_decimals = nb_decimals
        
        if self.position >= 0.6:
            self.reward = 1

        self.id = (round(self.velocity, nb_decimals),
                   round(self.position, nb_decimals))

        for action in [-1, 0, 1]:
            v, p = self.compute_next_state(action)
            self.add_transition((round(v, nb_decimals),
                                 round(p, nb_decimals)), action, 1)


    def next(self, action):
        v, p = self.compute_next_state(action)
        return MC_State_naive_discrete(v, p, self.nb_decimals)
            
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
             

# class MC_environment(Environment):
#     def __init__(self, init_state=None):
#         Environment.__init__(self, init_state)
