import matplotlib.pyplot as plt

import argparse

from laby.env import PDM_laby
import core.policy as policy
from core.rl import Agent
from core.rl import Trainer
from core.rl import Agent as SimpleAgent
import core.episode_runner as episode_runner
from core.estimator import HashTblEstimator
from laby.displayer import Log_displayer_nb_action, Displayer_evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--file", type=str)
ap.add_argument("--nb_iteration", type=int, default=100)
ap.add_argument("--learning_rate", type=float, default=0.8)
ap.add_argument("--discount", type=float, default=0.9)
args = vars(ap.parse_args())

env = PDM_laby(open(args['file']))

Y_alea = None
Y = None
X = None
episode_opti = None

estimator = HashTblEstimator()
policy = policy.N_Greedy(0.80)
# policy = policy.Simulated_Anealing_N_Greedy(0.90, 20, 0.98)
# policy = policy.N_freq_greedy(10)

agent = Agent(policy, estimator)

displayer_log = Log_displayer_nb_action(agent, env)
displayer_evaluate = Displayer_evaluate(None, env)
    
trainer_episode = episode_runner.Episode_train(agent, env, args['learning_rate'], args['discount'])
trainer = Trainer(agent, env, trainer_episode, displayer_evaluate)
trainer.train(args['nb_iteration'])
env.print_solution_agent(agent)
estimator.print()

# print("training done" + str(i))

# if episode_opti == None:
#     X, Y, Y_alea = displayer_log.get_result()
#     episode_opti = displayer_evaluate.get_result()
# else:
#     X_2, Y_2, Y_alea_2 = displayer_log.get_result()
#     episode_opti += displayer_evaluate.get_result()
#     Y += Y_2
#     Y_alea += Y_alea_2

# plt.plot(X, Y)
# # plt.plot(X, Y_alea)
# plt.plot([episode_opti, episode_opti], [0, max(Y)])
# plt.show()



# # agent = Agent(policy.N_Greedy(0.95))
# # # agent = Agent(policy.Simulated_Anealing_N_Greedy(0.90, 0.998))
# # # agent = Agent(policy.N_freq_greedy(10))
# # displayer_log = Log_displayer_nb_action(agent, env)
# # displayer_evaluate = Displayer_evaluate(None, env)
    
# # trainer_episode = episode_runner.Episode_train(agent, env, args['learning_rate'], args['discount'])
# # trainer = Trainer(agent, env, trainer_episode, displayer_evaluate)
# # trainer.train(args['nb_iteration'])
# # print("training done")

# # X, Y, Y_alea = displayer_log.get_result()
# # episode_opti = displayer_evaluate.get_result()
# # env.print_solution_agent(agent)
# # plt.plot(X, Y)
# # plt.plot(X, Y_alea)
# # plt.plot([episode_opti, episode_opti], [0, max(Y)])
# # plt.show()
