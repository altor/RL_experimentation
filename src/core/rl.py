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
        if self.debug:
            print("choosen action : " + str(action))
        return action
        
    def compute_q_value(self, state, reward, learning_rate, discount, actions):
        """
        :param action: possible actions in the state [state] 
                       (None my be an action)
        """
        if state.is_terminal:
            self.q_values[(state, None)] = reward
            actions.append(None)

        if self.previous_state != None:
            previous = (self.previous_state, self.previous_action)

            if previous in self.sa_frequency:
                self.sa_frequency[previous] += 1
            else:
                self.sa_frequency[previous] = 1
            
            # recherche de la meilleur valeur q(s', a')
            max_qvalue = -1000
            for action in actions:
                if (state,action) in self.q_values and self.q_values[(state, action)] > max_qvalue:
                    max_qvalue = self.q_values[(state, action)]
            if previous not in self.q_values:
                self.q_values[previous] = 0

            alpha = learning_rate / (learning_rate +
                                     self.sa_frequency[previous])
                
            self.q_values[previous] += (
                alpha * (
                    self.previous_reward + discount * max_qvalue
                    - self.q_values[previous]
                )
            )

        self.previous_state = state
        self.previous_reward = reward

class Trainer:
    def __init__(self, agent, environment, seed=None):
        self.agent = agent
        self.seed = seed
        self.environment = environment

    def train(self, nb_iteration, learning_rate, discount, first=True):
        # print("========================" + str(nb_iteration))
        if nb_iteration == 0:
            return
        if first:
            self.agent.q_values = {}

        # self.agent.sa_frequency = {}
        self.agent.new_episode()

            
        self.environment.re_init()
        state = self.environment.get_current_state()
        self.agent.compute_q_value(state, state.reward,
                                   learning_rate, discount,
                                   state.actions)

        while not state.is_terminal:
            action = self.agent.decide(state)
            state = self.environment.next_sa(action)
            reward = state.reward
            # print(state.id)
            self.agent.compute_q_value(state, reward, learning_rate,
                                       discount, state.actions)
        return self.train(nb_iteration - 1, learning_rate,
                          discount, False)

