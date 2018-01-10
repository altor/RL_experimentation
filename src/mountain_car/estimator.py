import numpy as np

from core.estimator import AbstractEstimator


class GridEstimator(AbstractEstimator):
    def __init__(self, x_len, y_len):
        self.x_len = x_len
        self.y_len = y_len
        self.estimator_is_present = None
        AbstractEstimator.__init__(self)

    def reinit(self):
        self.estimator =  {}
        self.estimator_is_present = {}

    def new_action_estimator(self, action):
        self.estimator[action] = np.zeros((self.x_len, self.y_len))
        self.estimator_is_present[action] = np.full((self.x_len,
                                                     self.y_len), False)

    def state_id_to_coord(self, state_id):
        v, p = state_id

        v2 = int((v + 0.07) * self.x_len / 0.14)
        p2 = int((p + 1.2) * self.y_len / 1.8)
        return v2, p2

    def contains(self, state_id, action_id):
        if action_id in self.estimator:
            coord = self.state_id_to_coord(state_id)
            return (self.estimator_is_present[action_id])[coord]
        return False

    def get_value(self, state_id, action_id):
        coord = self.state_id_to_coord(state_id)
        return (self.estimator[action_id])[coord]

    def set_value(self, state_id, action_id, q):
        coord = self.state_id_to_coord(state_id)
        if action_id not in self.estimator:
            self.new_action_estimator(action_id)

        (self.estimator_is_present[action_id])[coord] = True
        (self.estimator[action_id])[coord] = q
            
