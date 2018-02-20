from core.policy import Greedy
class StateLoop(Exception):
    pass

training_data = []


class Episode_runner:
    """
    Execute un episode avec une certaine politique
    """
    
    def __init__(self, agent, environment, policy=None):
        self.agent = agent
        self.policy = agent.policy if policy == None else policy
        self.environment = environment
        self.saved_policy = None

    def init_function(self):
        return 

    def function(self, state, reward, action):
        return True

    def compute_result(self):
        return -1

    def init_run(self):
        """ Fonction appelée au début de la fonction run
        """
        self.saved_policy = self.agent.policy
        self.agent.policy = self.policy

    def end(self):
        self.agent.policy = self.saved_policy

    def before(self):
        """ Fonction appelée avant que l'agent ne choisisse une action
        """

    def get_reward(self, state):
        return state.reward
    
    def run(self):
        self.init_run()
        self.init_function()
        self.agent.new_episode()
        self.environment.re_init()
        state = self.environment.get_current_state()
        self.function(state, state.reward, state.actions)
        while not state.is_terminal():
            self.before()
            action = self.agent.decide(state)
            state = self.environment.next_sa(action)
            reward = self.get_reward(state)
            if not self.function(state, reward, state.actions):
                self.agent.policy = self.saved_policy
                return -1

        training_data.append(self.environment.nb_step)
        self.end()
        return self.compute_result()

class Episode_train(Episode_runner):
    def __init__(self, agent, environment, learning_rate, discount):
        Episode_runner.__init__(self, agent, environment)
        self.learning_rate = learning_rate
        self.discount = discount

    def function(self, state, reward, actions):
        # print(str(state) + " " + str(reward) + " " + str(action))
        # print(str(self.learning_rate) + " " + str(self.discount))
        if self.environment.render_bool and self.agent.previous_state != None:
            # id = (self.agent.previous_state.id, self.agent.previous_action)
            txt1="TRAINING"
            # txt1 = (self.agent.q_values[id]
            #         if id in self.agent.q_values
            #         else "none")
            # txt2 = self.agent.policy.rendering_info
            self.environment.render(txt1, None, waitkey=False)

        self.agent.compute_q_value(state, reward, self.learning_rate,
                                   self.discount, state.actions)
        return True

    def end(self):
        Episode_runner.end(self)
        # print(str(self.environment.nb_step))


class Double_episode(Episode_runner):
    def __init__(self, agent1, agent2, environment, policy=None):
        Episode_runner.__init__(self, agent1, environment, policy)
        self.other_agent = agent2
        self.agent1 = agent1
        self.agent2 = agent2

    def init_run(self):
        Episode_runner.init_run(self)
        
    def before(self):
        Episode_runner.before(self)
        if self.environment.is_other_agent_turn():
            # print("CHAAAAANNGEMEEEEEENT !!!!!!")
            pivot = self.agent
            self.agent = self.other_agent
            self.other_agent = pivot

    def end(self):
        res = Episode_runner.end(self)
        self.agent = self.agent1
        self.other_agent = self.agent2
        return res
            
    def get_reward(self, state):
        Episode_runner.get_reward(self, state)
        if state.is_terminal():
            saved_agent = self.agent
            self.agent = self.other_agent

            r1, r2 = state.reward
            self.function(state, r1, state.actions)
            self.agent = saved_agent
            return r2
        return state.reward

class Double_episode_train(Double_episode):
    def __init__(self, agent1, agent2, environment, learning_rate, discount, policy=None):
        Double_episode.__init__(self, agent1, agent2, environment)
        self.learning_rate = learning_rate
        self.discount = discount

    def function(self, state, reward, action):
        # print(str(state) + "\n" + str(reward) + "\n" + str(action))
        # print("---------------")
        self.agent.compute_q_value(state, reward, self.learning_rate,
                                   self.discount, state.actions)
        return True

class Double_episode_greedy(Double_episode):
    def __init__(self, agent1, agent2, environment):
        Double_episode.__init__(self, agent1, agent2, environment, Greedy())
        
class Episode_greedy(Episode_runner):
    """
    Permet de faire tourner un episode avec un comportement greedy (pour évaluer la performance de la politique par exemple)
    """
    def __init__(self, agent, environment, max_step=None):
        Episode_runner.__init__(self, agent, environment, Greedy())
        self.visited_states = {}
        self.max_step = max_step
        self.step = 0
        
    def function(self, state, reward, actions):
        self.step +=1
        if state in self.visited_states:
            return False
        if self.step == self.max_step:
            return False
        Episode_runner.function(self, state, reward, actions)
        self.visited_states[state] = True
        return True
