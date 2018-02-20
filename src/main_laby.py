import matplotlib.pyplot as plt
import numpy as np

import argparse

from laby.env import PDM_laby
import core.policy as policy
from core.rl import Agent
from core.rl import Trainer
from core.rl import Agent as SimpleAgent
import core.episode_runner as episode_runner
from core.estimator import HashTblEstimator
from laby.displayer import Log_displayer_nb_action, Displayer_evaluate


"""
expe : [pdm_policy, pdm_value, q_learning]
"""

ap = argparse.ArgumentParser()
ap.add_argument("--file", type=str)
ap.add_argument("--nb_iteration", type=int, default=100)
ap.add_argument("--learning_rate", type=float, default=0.8)
ap.add_argument("--discount", type=float, default=0.9)
ap.add_argument("--expe", type=str)
ap.add_argument("--policy_name", type=str, default="EG")
ap.add_argument("--alpha_bool", type=bool, default=False)
ap.add_argument("--p1", type=float, default=0.2)
ap.add_argument("--p2", type=float, default=0)
ap.add_argument("--p3", type=float, default=0)
args = vars(ap.parse_args())

class PDM_agent:
    def __init__(self, policy):
        self.pdm_policy = policy
        self.policy = None
    def decide(self, state):
        return self.pdm_policy[state.id]
    def add_displayer(self, displayer):
        return None
    def new_episode(self):
        return None

def gen_policy(policy_name, p1=0, p2=0, p3=0):
    if policy_name == "SAEG":
        return policy.Simulated_Anealing_N_Greedy(p1, p2, p3)
    if policy_name == "SAEEG":
        return policy.Simulated_Anealing_episode_N_Greedy(p1, p2)
    if policy_name == "SAEETG":
        return policy.Simulated_Anealing_episode_threshold_N_Greedy(p1, p2, p3)
    if policy_name == "EG":
        return  policy.N_Greedy(p1)


def evaluate(agent, env, trainer, nb_iteration, displayer_log, displayer_evaluate):
    displayer_log.reinit()
    displayer_evaluate.reinit()
    trainer.train(nb_iteration)
    episode = episode_runner.Episode_greedy(agent, env, max_step=3000)
    episode.run()
    X, Y, Y_alea = displayer_log.get_result()
    episode_opti = displayer_evaluate.get_result()
    return X, Y, episode_opti

env = PDM_laby(open(args['file']))    
agent = None
X, Y = None, None
if args['expe'] == 'pdm_policy':
    agent = PDM_agent(env.policy_iteration(['S' for i in range(len(env.states))], args['discount']))
    X, Y = np.arange(len(env.convergence)), env.convergence

elif args['expe'] == 'pdm_value':
    agent = PDM_agent(env.value_iteration(args['discount'], 0.01))
    X, Y = np.arange(len(env.convergence)), env.convergence

elif args['expe'] == 'q_learning':
    policy = gen_policy(args['policy_name'], args['p1'], args['p2'], args['p3'])
    estimator = HashTblEstimator()
    agent = Agent(policy, estimator)
    displayer_log = Log_displayer_nb_action(agent, env)
    displayer_evaluate = Displayer_evaluate(None, env)
    trainer_episode = episode_runner.Episode_train(agent, env, args['learning_rate'], args['discount'])
    trainer = Trainer(agent, env, trainer_episode, displayer_evaluate)
    trainer.train(args['nb_iteration'])
    X, Y, Y_alea = displayer_log.get_result()
    # episode_opti = displayer_evaluate.get_result()
    # plt.plot([episode_opti, episode_opti], [0, max(Y)])

elif args['expe'] == 'evaluate':
    accX = np.zeros(args['nb_iteration'])
    accY = np.zeros(args['nb_iteration'])
    acc_episode_opti = 0
    pol = gen_policy(args['policy_name'], args['p1'], args['p2'], args['p3'])
    estimator = HashTblEstimator()
    for i in range(10):
        print(i)
        agent = Agent(pol, estimator, args['alpha_bool'])
        displayer_log = Log_displayer_nb_action(agent, env)
        displayer_evaluate = Displayer_evaluate(None, env)
        trainer_episode = episode_runner.Episode_train(agent, env, args['learning_rate'], args['discount'])
        trainer = Trainer(agent, env, trainer_episode, displayer_evaluate)
        X, Y, episode_opti = evaluate(agent, env, trainer, args['nb_iteration'], displayer_log, displayer_evaluate)
        accX += X
        accY += Y
        acc_episode_opti += episode_opti
    X = accX / 10
    Y = accY / 10
    # episode_opti = acc_episode_opti / 10
    # plt.plot([episode_opti, episode_opti], [0, max(Y)])
elif args['expe'] == 'alpha':
    policy = gen_policy(args['policy_name'], args['p1'], args['p2'], args['p3'])
    estimator = HashTblEstimator()
    X = []
    Y = []
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]:
        print("alpha : " + str(alpha))

        acc_episode_opti = 0
        for i in range(10):
            print(i)
            agent = Agent(policy, estimator, alpha_bool=True)
            displayer_log = Log_displayer_nb_action(agent, env)
            displayer_evaluate = Displayer_evaluate(None, env)
            trainer_episode = episode_runner.Episode_train(agent, env, alpha, args['discount'])
            trainer = Trainer(agent, env, trainer_episode, displayer_evaluate)
            _, _, episode_opti = evaluate(agent, env, trainer, args['nb_iteration'], displayer_log, displayer_evaluate)
            acc_episode_opti += episode_opti
        X.append(alpha)
        Y.append(acc_episode_opti / 10)
        print(acc_episode_opti / 10)

elif args['expe'] == 'epsilon_eg':

    X = []
    Y = []
    for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        print("epsilon : " + str(epsilon))
        pol = policy.N_Greedy(epsilon)
        estimator = HashTblEstimator()
        acc_episode_opti = 0
        for i in range(10):
            print(i)
            agent = Agent(pol, estimator, alpha_bool=False)
            displayer_log = Log_displayer_nb_action(agent, env)
            displayer_evaluate = Displayer_evaluate(None, env)
            trainer_episode = episode_runner.Episode_train(agent, env, args['learning_rate'], args['discount'])
            trainer = Trainer(agent, env, trainer_episode, displayer_evaluate)
            _, _, episode_opti = evaluate(agent, env, trainer, args['nb_iteration'], displayer_log, displayer_evaluate)
            acc_episode_opti += episode_opti
        X.append(epsilon)
        Y.append(acc_episode_opti / 10)
        print(acc_episode_opti / 10)
        

plt.plot(X, Y)
plt.show()
env.print_solution_agent(agent)
