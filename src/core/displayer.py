import sys

class Displayer:
    def __init__(self, agent, environment, output=sys.stdout):
        self.agent = agent
        if agent != None :
            self.agent.add_displayer(self)
        self.environment = environment
        self.output = output
        self.acc = self.init_acc()

    def init_acc(self):
        return None

    def acc_to_string(self, acc):
        return ""
    
    def display(self):
        print(self.acc_to_string(self.acc), file=self.output)

class Log_displayer(Displayer):
    def __init__(self, agent, environment, output=sys.stdout):
        """
        [(0, episode_id) | (1, (state, action, is_random))]
        TODO : acc = liste de liste avec une liste par episode, on ajoute un deuxième accumulateur qui récupère les info envoyées par notify lors de l'appel de la fonction end_episode, cet accumulateur est ajouté au vrai acc
        """
        Displayer.__init__(self, agent, environment, output)
        self.episode_acc = []
        self.current_episode = 0

    def init_acc(self):
        return [(0,0)]
        
    def acc_to_string(self, acc):
        s = ""
        for (id, val) in acc:
            if id == 0:
                s += "########## episode " + str(val) + "\n"
            elif id== 1:
                state, action, is_random = val
                s += str(state) + " " + str(action)
                if is_random:
                    s += " R\n"
                else:
                    s += " G\n"
        return s
        
    def end_episode(self):
        self.current_episode += 1
        self.acc.append((0, self.current_episode))

    def notify(self, state, action, q_value, is_random):
        self.acc.append((1, (state, action, is_random)))
                        
    
    
