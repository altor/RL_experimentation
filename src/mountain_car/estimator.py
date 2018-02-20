import numpy as np
import random

import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation

from core.estimator import AbstractEstimator
from mountain_car.continue_grid import Continue_grid
from sklearn.preprocessing import StandardScaler

zoom_bool=False
zoom=2
td_error_threshold=8
threshold=3
step=0.01
batch_size = 40
mem_lim = 80
mem_proba = 0.2
class GridEstimator(AbstractEstimator):
    def __init__(self, x_step, y_step, nb_td_error):
        self.nb_td_error = nb_td_error
        self.x_step = x_step
        self.y_step = y_step
        self.estimator_is_present = {}
        # contient, pour chaque couple case,action : nb_visite, sum erreur_td
        self.td_error_acc = {}
        AbstractEstimator.__init__(self)
            
    def reinit(self):
        self.estimator =  {}
        # x_step = 0.14 / self.x_len
        # y_step = 1.8 / self.y_len
        for action in [-1, 0 , 1, None]:
            self.estimator[action] = Continue_grid(-0.07, 0.07, -1.2,
                                                   0.6, self.x_step,
                                                   self.y_step,
                                                   (0, 0, 0))
        self.estimator_is_present = {}
        self.td_error_acc = {}
        
    # def state_id_to_coord(self, state_id):
    #     v, p = state_id

    #     v2 = int((v + 0.07) * self.x_len / 0.14)
    #     p2 = int((p + 1.2) * self.y_len / 1.8)
    #     return v2, p2

    def contains(self, state_id, action_id):
        # if action_id in self.estimator:
        #     coord = self.state_id_to_coord(state_id)
        #     return (self.estimator_is_present[action_id])[coord]
        return True

    def get_value(self, state_id, action_id):
        x,y = state_id
        q , _, _ = (self.estimator[action_id]).get(x, y)
        return q

    def set_value(self, state_id, action_id, q):
        x,y = state_id
        # if action_id not in self.estimator:
        #     self.new_action_estimator(action_id)

        # (self.estimator_is_present[action_id])[state_id] = True
        (self.estimator[action_id]).set(x, y, (q, 1, 0))

    def increase(self, state_id, action_id, td_error):
        # AbstractEstimator.increase(self, state_id, action_id, val)
        
        x,y = state_id
        # print(str(x) + " " + str(y))
        # if x > 0.021 and x < 0.0248 and y > -0.299 and y < -0.211:
        #     print(td_error)
        
        q , nb, td_error_sum = (self.estimator[action_id]).get(x, y)
        # new_q = q + td_error
        # moy = 0
        # for (x_dir, y_dir) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        #     x2, y2 = (x + x_dir * step), (y + y_dir * step)
        #     q2, _, _ = self.estimator[action_id].get(x2, y2)
        #     moy += abs(q2 - new_q)
        # if  moy / 4 > threshold:
        #     (self.estimator[action_id]).discretise(x, y, zoom, zoom)
        # new_val = (new_q, nb + 1, td_error_sum + abs(td_error))
        # (self.estimator[action_id]).set(x, y, new_val)
        
        if (td_error_sum >= td_error_threshold) and zoom_bool:
            new_val = (q + td_error, 0, abs(td_error))
            (self.estimator[action_id]).set(x, y, new_val)
            (self.estimator[action_id]).discretise(x, y, zoom, zoom)
        else:
            new_val = (q + td_error, nb + 1, td_error_sum + abs(td_error))
            (self.estimator[action_id]).set(x, y, new_val)

    def display_td_error_acc(self):
        error_indicators = {}
        for action in [-1, 0, 1]:
            data_list = []
            for (_, nb, td_error_sum) in self.estimator[action]:
                 data_list.append(td_error_sum)
            
            data = np.array(data_list)
            print(str(action) + " : " + str(np.mean(data)) + ";" + str(np.sqrt(np.var(data))))

class Neural_network_estimator(AbstractEstimator):
    def __init__(self):
        self.networks = []
        f = open('nn_data/scaler.dump', 'rb')
        self.scaler = pickle.load(f)
        for a in [-1, 0, 1]:
            model = Sequential([
                Dense(100, input_shape=(2,), init='lecun_uniform'),
                Activation('relu'),
                Dense(100, init='lecun_uniform'),
                Activation('relu'),
                Dense(100),
                Activation('relu'),
                Dense(100),
                Activation('relu'),
                Dense(100),
                Activation('relu'),
                Dense(100),
                Activation('relu'),
                Dense(1),
                Activation('linear'),
            ])

            model.compile(optimizer='rmsprop', loss='mse')
            model.load_weights('nn_data/nn_' + str(a) + '.dump',
                               by_name=False)
            self.networks.append(model)
        print('network ready')
        self.mem = []
        self.mem_target = []
        self.a = 0
    def reinit(self):
        return None

    def contains(self, state_id, action_id):
        return True

    def get_value(self, state_id, action_id):
        # self.a += 1
        # print(self.a)
        x,y = state_id
        # data = self.scaler.transform([[x, y]])
        data = self.scaler.transform([[x, y]])
        mlp = self.networks[action_id + 1]
        return (mlp.predict_on_batch(data))[0]

    def set_value(self, state_id, action_id, td_error):
        return None

    def add_to_mem(self, data, val):
        if len(self.mem_target) >= mem_lim:
            if random.random() < mem_proba:
                i = random.randint(0,len(self.mem_target) - 1)
                self.mem_target.pop(i)
                self.mem.pop(i)
        self.mem.append(data)
        self.mem_target.append(val)
    
    def increase(self, state_id, action_id, td_error):
        q = self.get_value(state_id, action_id)
        x,y = state_id

        mlp = self.networks[action_id + 1]
        data = np.array([[x, y]])
        target = np.array([q])
        mlp.train_on_batch(self.scaler.transform(data), target)
