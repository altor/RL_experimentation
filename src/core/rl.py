import sys

from core.episode_runner import Episode_greedy
from core.episode_runner import Episode_train
from core.episode_runner import Episode_runner

class Agent:

    def __init__(self, policy, debug=False):
        self.policy = policy
        self.debug = debug
        
        self.final_reward = None

        self.previous_state = None
        self.previous_reward = None
        self.previous_action = None
        self.q_values = {}
        self.sa_frequency = {}


    def add_displayer(self, displayer):
        self.policy.add_displayer(displayer)

    def remove_displayer(self, displayer):
        self.policy.remove_displayer(displayer)
        
    def new_episode(self):
        """
        notify the agent that a new episode begin

        empty the hashtable sa_frequency
        notify the policy
        """
        self.sa_frequency = {}
        self.policy.new_episode()
        
    def decide(self, state):
        action = self.policy.action(state, self.q_values,
                                    self.sa_frequency)
        self.previous_action = action
        # print("choosen action : " + str(action) + " state : " + str(state))
        return action
        
    def compute_q_value(self, state, reward, learning_rate, discount, agent_actions):
        """
        :param action: possible actions in the state [state] 
                       (None maybe an action)
        """
        actions = agent_actions.copy()
        if state.is_terminal:
            self.q_values[(state.id, None)] = reward
            actions.append(None)
        if self.previous_state != None:
            previous_id = (self.previous_state.id, self.previous_action)

            if previous_id in self.sa_frequency:
                self.sa_frequency[previous_id] += 1
            else:
                self.sa_frequency[previous_id] = 1
            
            # recherche de la meilleur valeur q(s', a')
            max_qvalue = -1000
            for action in actions:
                if (state.id, action) in self.q_values and self.q_values[(state.id, action)] > max_qvalue:
                    max_qvalue = self.q_values[(state.id, action)]
            if previous_id not in self.q_values:
                self.q_values[previous_id] = 0
            if max_qvalue == -1000:
                max_qvalue = 0

            alpha = learning_rate / (learning_rate +
                                     self.sa_frequency[previous_id])
            # print(max_qvalue)
            self.q_values[previous_id] += (
                alpha * (
                    self.previous_reward + discount * max_qvalue
                    - self.q_values[previous_id]
                )
            )

        self.previous_state = state
        self.previous_reward = reward

class Trainer:
    def __init__(self, agent, environment, episode_runner, displayer=None, seed=None):
        self.agent = agent
        self.seed = seed
        self.environment = environment
        self.displayer = displayer
        self.episode_runner = episode_runner

    def train(self, nb_iteration):
        self.agent.q_values = {}
        for i in range(nb_iteration):
            print(i)
            # print(self.agent.q_values)
            # print("====================\n\n\n")
            self.episode_runner.run()
            if self.displayer != None:
                displayer = self.displayer
                class Episode_evaluation(Episode_greedy):
                    def __init__(self, agent, environment):
                        Episode_greedy.__init__(self, agent, environment)
                    def init_run(self):
                        Episode_greedy.init_run(self)
                        #print(self.agent.policy)
                        self.agent.add_displayer(displayer)
                    def end(self):
                        self.agent.remove_displayer(displayer)

                episode = Episode_evaluation(self.agent,
                                             self.environment)
                episode.run()

        return
