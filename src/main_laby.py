from laby.env import PDM_laby
import core.policy as policy
from laby.rl import Agent
from core.rl import Trainer
from core.rl import Agent as SimpleAgent

env = PDM_laby(open("laby.1"))

# agent = Agent(policy.N_freq_greedy(10))
# agent = Agent(policy.Simulated_Anealing_N_Greedy(0.9, 0.998))
# agent = Agent(policy.N_Greedy(0.8))
# trainer = Trainer(agent, env)
# trainer.train(800, 0.8, 0.9)a
# print("training done")
# agent.policy = policy.Greedy()
# env.print_solution_agent(agent)


# policy = env.policy_iteration(env.random_policy(), 0.999)
policy = env.value_iteration(0.999, 0.0001)
env.print_solution_policy(policy)
