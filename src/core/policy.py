import random


class Virtual_Greedy:

    def __init__(self):
        self.displayers = []


    def add_displayer(self, displayer):
        self.displayers.append(displayer)
        
    def remove_displayer(self, displayer):
        if displayer in self.displayers:
            i = self.displayers.index(displayer)
            del self.displayers[i]
        self.displayers.append(displayer)
        
    def new_episode(self):
        for displayer in self.displayers:
            displayer.end_episode()
        return None
    
    def greedy_choice(self, state, q_values):
        actions = state.actions.copy()
        random.shuffle(actions)
        best_action = actions[0]
        best_action_value = -1000
        for action in state.actions:
            if (state.id, action) in q_values:
                if q_values[(state.id, action)] > best_action_value:
                    best_action = action
                    best_action_value = q_values[(state.id, action)]
        return best_action

    def random_choice(self, state):
        actions = state.actions
        i = random.randint(0, len(actions) - 1)
        return actions[i]

class Greedy(Virtual_Greedy):
    def __init__(self):
        Virtual_Greedy.__init__(self)
    
    def action(self, state, q_values, sa_frequency):
        action = self.greedy_choice(state, q_values)
        for displayer in self.displayers:
            displayer.notify(state, action, False)
        return action
    
    
class N_Greedy(Virtual_Greedy):
    def __init__(self, n):
        Virtual_Greedy.__init__(self)
        """
        :param n: 0 < n < 1 probability to choose a random choice
        """
        self.n = n

    def action(self, state, q_values, sa_frequency):

        if random.random() < self.n:
            action = self.random_choice(state)
            for displayer in self.displayers:
                displayer.notify(state, action, True)
            return action
        else:
            action = self.greedy_choice(state, q_values)
            for displayer in self.displayers:
                displayer.notify(state, action, False)
            return action
        

class Simulated_Anealing_N_Greedy(N_Greedy):
    def __init__(self, n, ratio):
        """
        :param ratio: at each new episode, the probability to choose a random choice decrease by ratio (0 < ratio < 1)
        """
        N_Greedy.__init__(self, n)
        self.ratio = ratio

    def new_episode(self):
        N_Greedy.new_episode(self)
        self.n *= self.ratio
        # print(self.n, end = " ")

        
class N_freq_greedy(Virtual_Greedy):
    def __init__(self, n):
        """
        :param n: number of time the action must be called before make a random choice
        """
        Virtual_Greedy.__init__(self)
        self.n = n

    def action(self, state, q_values, sa_frequency):
        action = self.greedy_choice(state, q_values)
        id = (state.id, action)
        if id in sa_frequency and sa_frequency[id] < self.n:
            for displayer in self.displayers:
                displayer.notify(state, action, False)
            return action
        random_action = self.random_choice(state)
        for displayer in self.displayers:
            displayer.notify(state, random_action, True)
        return random_action


