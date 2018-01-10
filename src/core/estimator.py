class AbstractEstimator:
    def __init__(self):
        self.estimator = None
        self.reinit()

    def reinit(self):
        return None

    def contains(self, state_id, action_id):
        return False
    
    def get_value(self, state_id, action_id):
        return None

    def set_value(self, state_id, action_id, q):
        return None

    def increase(self, state_id, action_id, val):
        init_val = self.get_value(state_id, action_id)
        self.set_value(state_id, action_id, init_val + val)

class HashTblEstimator(AbstractEstimator):
    def __init__(self):
        AbstractEstimator.__init__(self)

    def reinit(self):
        self.estimator = {}

    def contains(self, state_id, action_id):
        return (state_id, action_id) in self.estimator

    def get_value(self, state_id, action_id):
        if (state_id, action_id) in self.estimator:
            return self.estimator[(state_id, action_id)]
        return None

    def set_value(self, state_id, action_id, q):
        self.estimator[(state_id, action_id)] = q
