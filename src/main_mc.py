import argparse

import core.policy as policy
from core.rl import Agent, Trainer
from core.env import Environment 
import core.episode_runner as episode_runner
from laby.displayer import Log_displayer_nb_action
from core.displayer import Log_displayer
from mountain_car.env import MC_State_naive_discrete

ap = argparse.ArgumentParser()
ap.add_argument("--nb_iteration", type=int, default=100)
ap.add_argument("--learning_rate", type=float, default=0.8)
ap.add_argument("--discount", type=float, default=0.9)
ap.add_argument("--rnd", type=int, default=2)
args = vars(ap.parse_args())

rnd = args['rnd']
env = Environment(init_state = MC_State_naive_discrete(0, -0.5, rnd))
agent = Agent(policy.N_Greedy(0.98))
displayer_log = Log_displayer_nb_action(agent, env)
displayer_q = Log_displayer(agent, env, shape_state_space=(pow(10, rnd), int(round(pow(10, rnd) * 1.8))))
episode_runner = episode_runner.Episode_train(agent, env, args['learning_rate'], args['discount'])
trainer = Trainer(agent, env, episode_runner)
trainer.train(args['nb_iteration'])
print("training done")

displayer_log.display()
displayer_q.display_q_acc()
