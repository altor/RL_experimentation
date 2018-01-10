import random


class Virtual_Greedy:

    def __init__(self):
        self.displayers = []
        self.rendering_info = None


    def get_name(self):
        return ""
        
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
        # actions = state.actions.copy()
        # random.shuffle(actions)
        best_action = None
        best_action_value = -1000000
        for action in state.actions:
            if q_values.contains(state.id, action):
                if q_values.get_value(state.id, action) > best_action_value:
                    best_action = action
                    best_action_value = q_values.get_value(state.id,
                                                           action)

        if best_action == None:
            return self.random_choice(state)
        return best_action

    def random_choice_action_unvisited(self, state, sa_frequency):
        """
        fait un choix random parmis les actions qui n'ont pas encore été prise, si toute les action ont été prise, effectue un random choice normal
        """
        actions_unvisited = []

        for action in state.actions:
            if (state.id, action) not in sa_frequency:
                actions_unvisited.append(action)

        if actions_unvisited == []:
            return self.random_choice(state)

        else:
            i = random.randint(0, len(actions_unvisited) - 1)
            return actions_unvisited[i]
        
    
    def random_choice(self, state):
        actions = state.actions
        i = random.randint(0, len(actions) - 1)
        return actions[i]

class Greedy(Virtual_Greedy):
    def __init__(self):
        Virtual_Greedy.__init__(self)

    def get_name(self):
        return "Greedy"
        
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

    def get_name(self):
        return "N_greedy_" + str(self.n)
        
    def before_action(self, state):
        return None
    
    def action(self, state, q_values, sa_frequency):
        self.before_action(state)
        if random.random() < self.n:
            action = self.random_choice_action_unvisited(state,
                                                         sa_frequency)
            # action = self.random_choice(state)
            for displayer in self.displayers:
                displayer.notify(state, action, True)
            return action
        else:
            action = self.greedy_choice(state, q_values)
            for displayer in self.displayers:
                displayer.notify(state, action, False)
            return action
        

class Simulated_Anealing_episode_N_Greedy(N_Greedy):
    def __init__(self, n, ratio):
        """
        :param ratio: at each new episode, the probability to choose a random choice decrease by ratio (0 < ratio < 1)
        """
        N_Greedy.__init__(self, n)
        self.ratio = ratio

    def get_name(self):
        return "SANG_episode_" + str(self.n) + "_" + str(self.ratio)
        
    def new_episode(self):
        N_Greedy.new_episode(self)
        self.n *= self.ratio
        self.rendering_info = self.n
        # print(self.n, end = " ")

class Simulated_Anealing_N_Greedy(N_Greedy):
    """
    at each t step, the probability to choose a random choice decrease by ratio (0 < ratio < 1)
    """
    def __init__(self, n, t, ratio):
        N_Greedy.__init__(self, n)
        self.t = t
        self.ratio = ratio
        self.i = 0
        self.init_n = n
        self.rendering_info = n

    def get_name(self):
        return "SANG_" + str(self.n) + "_" + str(self.ratio) + "_" + str(self.t)
        
    def new_episode(self):
        N_Greedy.new_episode(self)
        self.n = self.init_n
        
    def before_action(self, state):
        self.i += 1
        if self.i == self.t:
            self.i = 0
            self.n *= self.ratio
            self.rendering_info = self.n

class N_freq_Simulated_anealing_greedy(N_Greedy):
    def __init__(self, n, t, ratio):
        """
        :param t: number of time the state is reached before make decrease the probability to make a random choice
        """
        N_Greedy.__init__(self, n)
        self.n = n
        self.visited_states = {}
        self.random_proba = {}
        self.ratio = ratio
        self.init_n = n
        self.t = t

    def get_name(self):
        return "SANFG_" + str(self.n) + "_" + str(self.ratio) + "_" + str(self.t)
        
    def before_action(self, state):
        if state.id not in self.visited_states:
            self.visited_states[state.id] = 0
            self.random_proba[state.id] = self.init_n
        self.visited_states[state.id] += 1

        if self.visited_states[state.id] > self.t:
            self.visited_states[state.id] = 0
            self.random_proba[state.id] *= self.ratio

        self.n = self.random_proba[state.id]
        self.rendering_info = self.n
        
