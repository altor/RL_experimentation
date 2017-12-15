import core.policy as policy
from core.displayer import Displayer
import core.rl
import sys


# class Result_displayer(Displayer):
#     """
#     Accumule les résultat de l'agent
#     :attribute result: List containing agent score after n episodes
#     :attribute acc: accumulateur utilisé pour calculer le score de l'agent lors d'un épisode 
#     """
#     def __init__(self, agent, environment, output=sys.stdout):
#         Displayer.__init__(self, agent, environment, output)
#         self.result = []

#     def append_acc(self, s, r, a):
#         return

#     def compute_acc(self, environment):
#         return acc
        
#     def evaluate(self):
#         displayer = self

#         class Episode_evaluate (Episode_greedy):
#             def __init__(self, agent, environment) :
#                 Episode_greedy.__init__(self, agent, environment)
#                 displayer.init_acc()
#             def init_function(self):
#                 Episode_greedy.init_function(self)
#                 displayer.init_acc()
#             def function(self, s, r, a):
#                 result = Episode_greedy.function(self, s, r, a)
#                 displayer.append_acc(s,r,a)
#                 return result
#             def compute_result(self, env=self.environment):
#                 return displayer.compute_acc(self.environment)

#         self.result.append(
#             Episode_evaluate(self.agent, self.environment).run()
#         )

#     def display(self):
#         print ("optimal reached after " + str(self.result.index(self.environment.max_score)) + " episodes")
#         for i in range(len(self.result)):
#             print(str(i) + " " + str(self.result[i]), file=self.output)


# class Result_displayer(Result_displayer):
#     def __init__(self, agent, environement, output=sys.stdout):
#         Result_displayer.__init__(self, agent, environement,
#                                           output)
#     def init_acc(self):
#         self.acc = 0

#     def append_acc(self, s, r, a):
#         self.acc += 1

#     def compute_acc(self, env):
#         return self.acc
