import sys
import random
from core.episode_runner import Episode_greedy
from core.episode_runner import Episode_train
from core.episode_runner import Episode_runner

class Agent:

    def __init__(self, policy, estimator, debug=False):
        self.policy = policy
        self.debug = debug
        
        self.final_reward = None

        self.previous_state = None
        self.previous_reward = None
        self.previous_action = None
        self.q_values = estimator
        self.sa_frequency = {}

    def add_displayer(self, displayer):
        self.policy.add_displayer(displayer)

    def remove_displayer(self, displayer):
        self.policy.remove_displayer(displayer)

    def render_q_values(self):
        return None
        
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
        return action
        
    def compute_q_value(self, state, reward, learning_rate, discount, agent_actions):
        """
        :param action: possible actions in the state [state] 
                       (None maybe an action)
        """
        actions = agent_actions.copy()
        random.shuffle(actions)
        if state.is_terminal():
            self.q_values.set_value(state.id, 0, reward)
            # self.q_values.get_value(state.id, None, reward)
            # actions.append(None)
        if self.previous_state != None:
            previous_id = (self.previous_state.id, self.previous_action)

            if previous_id in self.sa_frequency:
                self.sa_frequency[previous_id] += 1
            else:
                # print(str(len(self.sa_frequency)) + " " + str(previous_id))
                self.sa_frequency[previous_id] = 1
            
            # recherche de la meilleur valeur q(s', a')
            max_qvalue = -1000
            for action in actions:
                if self.q_values.contains(state.id, action):
                    # self.q_values.set_value(state.id, action, 0)
                    if self.q_values.get_value(state.id, action) > max_qvalue:
                        max_qvalue = self.q_values.get_value(state.id, action)
                        
            if not self.q_values.contains(self.previous_state.id,
                                          self.previous_action):
                self.q_values.set_value(self.previous_state.id,
                                        self.previous_action, 0)
            if max_qvalue == -1000:
                max_qvalue = 0

            alpha = learning_rate / (learning_rate +
                                     self.sa_frequency[previous_id])

            # print(str(alpha))

            td_error = (self.previous_reward + discount * max_qvalue
                        - self.q_values.get_value(self.previous_state.id,
                                                  self.previous_action))

            for displayer in self.policy.displayers:
                displayer.notify_td_error(self.previous_state, self.previous_action, td_error)
            
            # print(max_qvalue)
            self.q_values.increase(self.previous_state.id,
                                   self.previous_action,
                                   alpha * td_error)

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
        self.agent.q_values.reinit()
        for i in range(nb_iteration):
            # print(i, end=' ')
            # print(self.agent.q_values)
            # print("====================\n\n\n")
            self.episode_runner.run()
            if self.displayer != None:
                displayer = self.displayer
                class Episode_evaluation(Episode_greedy):
                    def __init__(self, agent, environment):
                        Episode_greedy.__init__(self, agent,environment, max_step=3000 )
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
