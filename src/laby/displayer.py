import matplotlib.pyplot as plt
import sys
from core.displayer import Log_displayer
from core.episode_runner import Episode_greedy
import numpy as np
class Displayer_evaluate(Log_displayer):
    def __init__(self, agent, environment, output=sys.stdout):
        """
        évalue le nombre d'épisode nécessaire pour obtenir la politique optimale
        acc : [nb_action_1, ..., nb_action_n]
        """
        Log_displayer.__init__(self, agent, environment, output)
        self.episode_acc = 0
        
    def init_acc(self):
        return []

    def acc_to_string(self, acc):
        # print(acc)
        if self.environment.max_score in self.acc:
            episode = self.acc.index(self.environment.max_score)
            return str(episode)
        return "-1"

    def end_episode(self):
        self.acc.append(self.episode_acc)
        self.episode_acc = 0

    def get_result(self):
        if self.environment.max_score in self.acc:
            episode = self.acc.index(self.environment.max_score)
            return episode
        return -1

    def notify(self, state, action, is_random):
        self.episode_acc += 1

class Log_displayer_nb_action(Log_displayer):
    def __init__(self, agent, environment, output=sys.stdout):
        """
        sauvegarde le nombre d'actions effectuées par épisode et le nombre d'actions aléatoires effectuées par épisodes
        indique aussi les changement de politique
        acc : {alea = [nb_action_alea_1, ..., nb_action_alea_n]
               actions = [nb_action_1, ..., nb_action_n]
               greedy = [ [actions_1], ... ,[actions_n]}
        """
        Log_displayer.__init__(self, agent, environment, output)
        
    def init_acc(self):
        acc = {}
        acc['alea'] = []
        acc['actions'] = []
        acc['policy'] = []
        self.episode_alea = 0
        self.episode_actions = 0
        return acc

    def end_episode(self):
        self.acc['alea'].append(self.episode_alea)
        self.acc['actions'].append(self.episode_actions)
        self.episode_alea = 0
        self.episode_actions = 0

        class Episode_evaluation(Episode_greedy):
            def __init__(self, agent, environment):
                Episode_greedy.__init__(self, agent, environment)
                self.acc = 0
            def function(self, s, r, a):
                r = Episode_greedy.function(self, s, r, a)
                self.acc += 1
                return r

            def compute_result(self):
                return self.acc


        episode = Episode_evaluation(self.agent, self.environment)
        self.acc['policy'].append(episode.run())
        

    def notify(self, state, action, q_value, is_random):
        self.episode_actions += 1
        if is_random:
            self.episode_alea += 1

    def get_result(self):
        X = np.array(range(len(self.acc['actions'])))
        Y = np.array(self.acc['actions'])
        Y_alea = np.array(self.acc['alea'])
        return X, Y, Y_alea
            
    def display(self):

        # on cherche les episode ou l'agent change de politque optimale

        # Tableau Yi = 0 si l'episode i a la meme politique que
        # l'episode i-1, Yi = 1 sinon

        X = [0]
        Y = [1]
        old_policy, *policies = self.acc['policy']
        for i in range(len(policies)):
            if policies[i] != old_policy:
                X.append(i)
                Y.append(1)
                old_policy = policies[i]

        plt.scatter(X, Y)
        plt.plot(range(len(self.acc['actions'])), self.acc['actions'])
        plt.plot(range(len(self.acc['actions'])), self.acc['alea'])
        plt.show()
        

                
