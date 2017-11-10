import core.policy as policy
import core.rl

class Agent(core.rl.Agent):
    def __init__(self, policy, debug=False, traceFile=None):
        core.rl.Agent.__init__(self, policy, debug)
        self.traceFile = traceFile
        self.nb_step = 0
        self.nb_episode = 0

    def new_episode(self):
        """
        write in the traceFile or print the number of steap to reach the exit
        """
        core.rl.Agent.new_episode(self)
        string = str(self.nb_episode) + " " + str(self.nb_step)
        
        if self.traceFile == None:
            print(string)
        else:
            self.traceFile.write(string)
        self.nb_step = 0
        self.nb_episode += 1
        
    def decide(self, state):
        action = core.rl.Agent.decide(self, state)
        self.nb_step += 1
        return action
    
