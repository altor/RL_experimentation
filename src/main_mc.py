import argparse
import sys
import numpy as np

import core.policy as policy
from core.rl import Agent, Trainer
import core.episode_runner as episode_runner
from core.displayer import Log_displayer
from core.estimator import HashTblEstimator

from laby.displayer import Log_displayer_nb_action

from mountain_car.env import MC_State_naive_discrete
from mountain_car.env import MC_gym_environment 
from mountain_car.estimator import GridEstimator
from mountain_car.estimator import Neural_network_estimator

ap = argparse.ArgumentParser()
ap.add_argument("--nb_iteration", type=int, default=100)
ap.add_argument("--learning_rate", type=float, default=0.8)
ap.add_argument("--discount", type=float, default=0.9)
ap.add_argument("--v_rnd", type=int, default=20)
ap.add_argument("--p_rnd", type=int, default=20)
ap.add_argument("--render", type=bool, default=False)
ap.add_argument("--max_step", type=int, default=200)
ap.add_argument("--nb_td_error", type=int, default=10)
ap.add_argument("--action", type=str, default="demo")
ap.add_argument("--evaluate", type=bool, default=False)
ap.add_argument("--estimator", type=str, default='grid')


ap.add_argument("--policy_name", type=str, default="EG")
ap.add_argument("--p1", type=float, default=0.2)
ap.add_argument("--p2", type=float, default=0)
ap.add_argument("--p3", type=float, default=0)



args = vars(ap.parse_args())

grid_shape = (args['v_rnd'], args['p_rnd'])
env = MC_gym_environment(grid_shape, render_bool=args['render'], max_step=args['max_step'])
# estimator = HashTblEstimator()



def gen_estimator(estimator_name):
    if estimator_name == 'grid':
        return GridEstimator(args['v_rnd'], args['p_rnd'], args['nb_td_error'])
    elif estimator_name == 'nn':
        return Neural_network_estimator()

estimator = gen_estimator(args['estimator'])


def gen_policy(policy_name, p1=0, p2=0, p3=0):
    if policy_name == "SAEG":
        return policy.Simulated_Anealing_N_Greedy(p1, p2, p3)
    if policy_name == "SAEEG":
        return policy.Simulated_Anealing_episode_N_Greedy(p1, p2)
    if policy_name == "EG":
        return  policy.N_Greedy(p1)
    
policy = gen_policy(args['policy_name'], args['p1'], args['p2'], args['p3'])
agent = Agent(policy, estimator) 
episode = episode_runner.Episode_train(agent, env, args['learning_rate'], args['discount'])
trainer = Trainer(agent, env, episode)# , displayer_log)

def evaluate(agent, env, trainer, nb_iteration):
    trainer.train(nb_iteration)
    episode = episode_runner.Episode_greedy(agent, env, max_step=3000)
    episode.run()

    if env.nb_step >= 1500:
        print("PAS DE POLITIQUE OPTI", file=sys.stderr)
        return evaluate(agent, env, trainer, nb_iteration)
    else:
        return env.nb_step

if args['action'] == "evaluate":
    print(policy.get_name(), file=sys.stderr)
    acc = np.zeros(10)
    for i in range(10):
        print(i, file=sys.stderr)
        acc[i] = evaluate(agent, env, trainer, args['nb_iteration'])
    print(policy.get_name() +";" + str(args['learning_rate']) + ";" + str(args['discount']) + ";" + str(acc.mean()))

elif args['action'] == "demo":
    trainer.train(args['nb_iteration'])
    print("training done")
    # for i in episode_runner.training_data:
    #     print(i)

    estimator.display_value_function(0.002, 0.02)

    episode = episode_runner.Episode_greedy(agent, MC_gym_environment(grid_shape, render_bool=True, max_step=1000))
    episode.run()

elif args['action'] == "extract":
    trainer.train(args['nb_iteration'])

    data_files = []
    target_files = []

    for a in [-1, 0, 1]:
        data_files.append(open('data' + str(a) + '.txt', 'w'))
        target_files.append(open('target' + str(a) + '.txt', 'w'))

    v_axis = np.arange(-0.07, 0.07, 0.001)
    p_axis = np.arange(-1.2, 0.6, 0.01)

            
    for v in v_axis:
        for p in p_axis:
            for a in [-1, 0, 1]:
                data_files[a + 1].write(str(v) + ' ' + str(p) + '\n')
                target_files[a + 1].write(str(estimator.get_value((v, p), a)) + '\n')

    for a in [-1, 0, 1]:
        data_files[a + 1].close()
        target_files[a + 1].close()
