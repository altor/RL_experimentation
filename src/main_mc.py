import argparse

import core.policy as policy
from core.rl import Agent, Trainer
from mountain_car.env import MC_gym_environment 
import core.episode_runner as episode_runner
from laby.displayer import Log_displayer_nb_action
from core.displayer import Log_displayer
from mountain_car.env import MC_State_naive_discrete
from core.estimator import HashTblEstimator
from mountain_car.estimator import GridEstimator

ap = argparse.ArgumentParser()
ap.add_argument("--nb_iteration", type=int, default=100)
ap.add_argument("--learning_rate", type=float, default=0.8)
ap.add_argument("--discount", type=float, default=0.9)
ap.add_argument("--v_rnd", type=int, default=20)
ap.add_argument("--p_rnd", type=int, default=20)
ap.add_argument("--render", type=bool, default=False)
ap.add_argument("--max_step", type=int, default=200)
args = vars(ap.parse_args())

grid_shape = (args['v_rnd'], args['p_rnd'])
env = MC_gym_environment(grid_shape, render_bool=args['render'], max_step=args['max_step'])
# estimator = HashTblEstimator()
estimator = GridEstimator(args['v_rnd'], args['p_rnd'])


# env = Environment(init_state = MC_State_naive_discrete(0, -0.5, rnd), estimator)
agent = Agent(policy.Simulated_Anealing_N_Greedy(0.99, 100, 0.99), estimator)
# agent = Agent(policy.Simulated_Anealing_episode_N_Greedy(0.99, 0.99), estimator)
# agent = Agent(policy.N_Greedy(0.60), estimator)
# agent = Agent(policy.N_freq_Simulated_anealing_greedy(0.95, 50, 0.99), estimator)
displayer_log = Log_displayer_nb_action(agent, env)
displayer_q = Log_displayer(agent, env, shape_state_space=grid_shape)
episode = episode_runner.Episode_train(agent, env, args['learning_rate'], args['discount'])
trainer = Trainer(agent, env, episode)
trainer.train(args['nb_iteration'])
print("training done")

# displayer_log.display()
displayer_q.display_state_grid()
displayer_q.display_q_acc()

episode = episode_runner.Episode_greedy(agent, MC_gym_environment(grid_shape, render_bool=True, max_step=10000000))

episode.run()
