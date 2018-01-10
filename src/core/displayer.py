import matplotlib.pyplot as plt

import numpy as np
import sys

class Log_displayer:
    def __init__(self, agent, environment, output=sys.stdout, shape_state_space=1):
        """
        [[(state, action, is_random)] (une liste par episode)]
        TODO : acc = liste de liste avec une liste par episode, on ajoute un deuxième accumulateur qui récupère les info envoyées par notify lors de l'appel de la fonction end_episode, cet accumulateur est ajouté au vrai acc
        """
        self.agent = agent
        if agent != None :
            self.agent.add_displayer(self)
        self.environment = environment
        self.output = output
        self.acc = self.init_acc()

        self.acc_episode = self.init_acc_episode()
        self.current_episode = 0
        self.acc_q = {}
        self.state_grid = np.zeros(shape_state_space)
        self.shape_state_space = shape_state_space

    def init_acc(self):
        return []

    def init_acc_episode(self):
        return []
        
    def acc_to_string(self):
        s = ""
        for i in length(self.acc):
            s += "########## episode " + str(i) + "\n"
            for state, action, is_random in self.acc[i]:
                s += str(state) + " " + str(action)
                if is_random:
                    s += " R\n"
                else:
                    s += " G\n"
        return s

    def append_episode_to_acc(self):
        self.acc.append(self.acc_episode)
    
    def end_episode(self):
        self.append_episode_to_acc()
        self.acc_episode = self.init_acc_episode()

    def display(self):
        print(self.acc_to_string(), file=self.output)

    def display_q_acc(self):
        graph = {}

        error_indicators = {}
        
        for ((state, action), (nb, td_error_acc)) in self.acc_q.items():
            if not action in graph:
                graph[action] = np.zeros(self.shape_state_space)
                error_indicators[action] = []

            (graph[action])[state.to_space_grid_coord()] = td_error_acc / nb
            error_indicators[action].append(td_error_acc / nb)

        for (action, array) in graph.items():
            fig, ax = plt.subplots()
            pos = ax.imshow(array, cmap='Greens', origin='lower', extent=[-0.07,0.07,-1.2,0.6])
            fig.colorbar(pos, ax=ax)
            ax.set_aspect('auto', adjustable='box')
    
            plt.show()

        for (action, data_list) in error_indicators.items():
            data = np.array(data_list)
            print(str(action) + " : " + str(np.mean(data)) + ";" + str(np.var(data)))

    def display_state_grid(self):
        fig, ax = plt.subplots()
        pos = ax.imshow(self.state_grid, cmap='Blues', origin='lower', extent=[-0.07,0.07,-1.2,0.6])
        fig.colorbar(pos, ax=ax)
        ax.set_aspect('auto', adjustable='box')
        # plt.savefig(self.agent.policy.get_name() + '.png')
        plt.show()
        
    def notify(self, state, action, is_random):
        self.state_grid[state.to_space_grid_coord()] += 1
        self.acc_episode.append((state, action, is_random))

    def notify_td_error(self, state, action, td_error):
        if (state, action) in self.acc_q:
            n, td_error_acc = self.acc_q[(state, action)]
            self.acc_q[(state, action)] = (n + 1, td_error_acc + td_error)
        else:
            self.acc_q[(state, action)] = (1, td_error)
