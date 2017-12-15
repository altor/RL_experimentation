import numpy as np

from core.env import Environment, Markovian_State

class MC_State_naive_discrete(Markovian_State):
    def __init__(self, init_position, init_velocity, nb_decimals):
        """
        représentation discrète d'un état pour le problème de la voiture dans la montagne.
        La vitesse et la position sont arrondies à un nombre de décimale données

        :param nb_decimals: nombre de décimales auxquelles sont arondie 
                            les valeurs de l'état
        
        """
        Markovian_State.__init__(self, -1, -1)

        self.velocity = init_position
        self.position = init_velocity
        self.nb_decimals = nb_decimals
        
        if self.position >= 0.6:
            self.reward = 1

        self.id = (np.round(self.velocity, decimals=nb_decimals),
                   np.round(self.position, decimals=nb_decimals))

        for action in [-1, 0, 1]:
            self.add_transition(self.compute_next_state(action),
                                action, 1)


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
             np.cos(3 * self.position) * (-0.0025))
        return (v, self.position + v)
             

# class MC_environment(Environment):
#     def __init__(self, init_state=None):
#         Environment.__init__(self, init_state)
