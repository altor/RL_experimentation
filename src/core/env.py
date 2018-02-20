import numpy as np
from copy import deepcopy
import random

def compute_value(s, action, discount, state_values):
    val = 0
    for s_id in s.next_states(action):
        val += s.get_transition_probability(s_id, action) * state_values[s_id]
    return val * discount


class Environment:
    def __init__(self, init_state=None):
        self.current_state = deepcopy(init_state)
        self.init_state = init_state
        self.render_bool=False
        
    def get_current_state(self):
        return self.current_state
        
    def re_init(self):
        self.current_state = deepcopy(self.init_state)


    def next_sa(self, action):
        """
        renvoi l'état suivant et met a jour l'état courant
        """
        self.current_state = self.current_state.next(action)
        return self.current_state

class Double_Environment(Environment):
    """
    Environement ou deux agents peuvent intervenir
    possède une fonction permetant de dire quand le prochain agent doit agir
    """
    
    def __init__(self, init_state=None):
         Environment.__init__(self, init_state)
    def is_other_agent_turn(self):
        """
        Renvoi True si l'agent 
        """
        return False
    
class PDM(Environment):

    def __init__(self):
        self.convergence = []
        Environment.__init__(self)
        self.states = []
        self.policy = []
        self.current_state_id = 0
        self.final_state_id = -1
        self.mem = 0
        self.nb_step = 0
                
        
    def add_state(self, state):
        if self.init_state == None:
            self.init_state = deepcopy(state)
            self.current_state = state
        state.id = len(self.states)
        self.states.append(state)

    def set_policy(self, policy):
        if len(policy) != len(self.states):
            raise PolicyBadSize
        self.policy = policy

    def random_policy(self):
        policy = []
        for state in self.states:
            policy.append(state.actions[0])
        return policy

    def next_sa(self, action):
        id = self.current_state_id
        self.current_state_id = self.states[id].next_id(action)
        self.current_state = self.states[self.current_state_id]
        self.nb_step += 1
        return self.states[self.current_state_id]
    
    def next(self):
        id = self.current_state_id
        action = self.policy[id]
        return self.next_sa(action)

    def re_init(self):
        Environment.re_init(self)
        self.current_state_id = 0
        self.nb_step = 0

    def get_current_state(self):
        return self.states[self.current_state_id]
        
    def value_iteration(self, discount, threshold):
        # Calcul de la valeur optimale
        state_values = np.array([ 0.0 for _ in self.states])
        new_state_values = np.array([ 0.0 for _ in self.states])

        while True:
            self.mem += 1
            for i in range(len(state_values)):
                s = self.states[i]
                new_state_values[i] = s.reward
                max_val = -1
                for action in s.actions:
                    val = compute_value(s, action, discount, state_values)
                    if val > max_val:
                        max_val = val

                new_state_values[i] += max_val
            self.convergence.append((new_state_values - state_values).mean())
            if ((new_state_values - state_values) < threshold).all():
                break
            state_values = new_state_values.copy()
        print(self.mem)
        # Création du plan optimal
        policy = [None for _ in self.states]
        
        for i in range(len(state_values)):
            s = self.states[i]
            max_val = -1
            max_action = None
            for action in s.actions:
                val = compute_value(s, action, discount, state_values)
                if val > max_val:
                    max_val = val
                    max_action = action
            # policy[i] = (i, max_action)
            policy[i] = max_action

        return policy
    
    def policy_iteration(self, policy, discount):
        
        # Résolution des équations
        values = self.policy_value(policy, discount)
        # Amélioration de la politique
        new_policy = []

        for state in self.states:
            max_val = values[state.id]
            max_action = policy[state.id]
            for action in state.actions:
                val = state.reward
                val += compute_value(state, action, discount, values)

                if max_val < val:
                    max_val = val
                    max_action = action
            
            new_policy.append(max_action)
        if new_policy == policy:
            print(self.mem)
            return policy
        self.mem += 1
        return self.policy_iteration(new_policy, discount)

    def policy_value_iteration(self, initial_state_id, discount, threshold):
        current_values = np.zeros(len(self.states))
        new_values = np.zeros(len(self.states))

        while True:
            for state in self.states:
                i = state.id
                new_values[i] = state.reward
                if i != self.final_state_id:
                    new_values[i] += compute_value(state,
                                                   self.policy[i],
                                                   discount,
                                                   current_values)
            if ((new_values - current_values) < threshold).all():
                return new_values[initial_state_id]

            current_values = new_values.copy()


    def policy_value(self, policy, discount):
        equations = []
        result = []
        for state in self.states:
            result.append(state.reward)
            e = [0 for _ in self.states]
            e[state.id] = 1
            action = policy[state.id]

            if state.id != self.final_state_id:
                for next_state_id in state.next_states(action):
                    e[next_state_id] -= discount * state.get_transition_probability(next_state_id, action)

            equations.append(e)

        values = np.linalg.solve(np.array(equations), np.array(result))
        return values


    
    def policy_value_mc(self, initial_state_id, discount, nb_episode):
        # renvois la listes des états traversé lors de l'épisode
        def aux(state, acc, pow_discount):

            if self.final_state_id == state.id:
                return acc

            s2 = self.next()
            return aux(s2, acc + s2.reward * pow_discount,
                       discount * pow_discount)


        v = np.zeros(nb_episode)
        for i in range(nb_episode):
            self.current_state_id = initial_state_id
            s = self.states[initial_state_id]
            v[i] = aux(s, s.reward, discount)

        return np.average(v)

class State:
    def __init__(self, reward, actions=None, is_terminal_bool=False):
        self.reward = reward
        self.is_terminal_bool = is_terminal_bool
        self.actions = actions

    def next(self):
        return None
        
    def is_terminal(self):
        # print("tutu")
        return self.is_terminal_bool

    def to_space_grid_coord(self):
        return self.id
        
class Markovian_State(State):
    
    def __init__(self, id, reward):
        State.__init__(self, reward, actions=[])
        self.id = id
        self.transitions = []
    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)
    
    def add_transition(self, target_state_id, action_id, probability):
        if action_id not in self.actions:
            self.actions.append(action_id)

        self.transitions.append((target_state_id, action_id, probability))

    def get_transition_probability(self, target_state_id, target_action_id):
        for (state_id, action_id, probability) in self.transitions:
            if state_id == target_state_id and action_id == target_action_id:
                return probability


    """
    renvoi la liste des états ateignable par l'action action_id
    """
    def next_states(self, action_id):
        l = []
        for (s_id, a_id, proba) in self.transitions:
            if a_id == action_id:
                l.append(s_id)
        return l
    
    def next_id(self, action_id):
        if action_id == None:
            return self.id
            
        rank = random.random()
        l = []
        for (s_id, a_id, proba) in self.transitions:
            if a_id == action_id:
                l.append((s_id, proba))
        p = 0
        for (s_id, proba) in l:
            p += proba
            if rank < p:
                return s_id

        raise ValueError
