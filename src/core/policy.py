import random


class Virtual_Greedy:
    # def __init__(self):
    #     self.a=None

    def new_episode(self):
        return None
    
    def greedy_choice(self, state, q_values):
        actions = state.actions.copy()
        random.shuffle(actions)
        best_action = actions[0]
        best_action_value = -1000
        for action in state.actions:
            if (state, action) in q_values:
                if q_values[(state, action)] > best_action_value:
                    best_action = action
                    best_action_value = q_values[(state, action)]
        return best_action

    def random_choice(self, state):
        actions = state.actions
        i = random.randint(0, len(actions) - 1)
        return actions[i]


class Greedy(Virtual_Greedy):
    def action(self, state, q_values, sa_frequency):
        return self.greedy_choice(state, q_values)
    
    
class N_Greedy(Virtual_Greedy):
    def __init__(self, n):
        """
        :param n: 0 < n < 1 probability to choose a random choice
        """
        self.n = n

    def new_episode(self):
        print(self.n, end = " ")
        
    def action(self, state, q_values, sa_frequency):

        if random.random() < self.n:
            # print(str(self.n) + " random")
            return self.random_choice(state)
        else:
            # print(str(self.n) + " greedy")
            return self.greedy_choice(state, q_values)
        

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
        self.n = n

    def action(self, state, q_values, sa_frequency):
        action = self.greedy_choice(state, q_values)
        id = (state, action)
        if id in sa_frequency and sa_frequency[id] < self.n:
            return action
        return self.random_choice(state)
