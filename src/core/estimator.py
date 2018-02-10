import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

    def display_value_function(self, precision_v, precision_p):
        shape = (int(np.around(1.8/precision_p)), int(np.around(0.14 / precision_v)))
        val = np.zeros(shape)

        v_axis = np.arange(-0.07, 0.07, precision_v)
        p_axis = np.arange(-1.2, 0.6, precision_p)
        x = []
        y = []
        z = []
        # boolean = True
        for v in v_axis:
            for p in p_axis:
                max_q = -10000
                for a in [-1, 0, 1]:
                    q = self.get_value((v, p), a)
                    max_q = q if q > max_q else max_q
                i_v = int(np.around((v + 0.07) / precision_v))
                i_p = int(np.around((p + 1.2) / precision_p))
                val[i_p][i_v] = max_q
                x.append(p)
                y.append(v)
                z.append(max_q)

        fig = plt.figure()
        ax = Axes3D(fig)
        x1_len, x2_len = shape
        x1range = np.linspace(-1.2, 0.5, x1_len)
        x2range = np.linspace(-0.07, 0.07, x2_len)
        X, Y = np.meshgrid(x1range,x2range)
        # print(str(X.shape) + " " + str(Y.shape) + " " + str(val.T.shape))
        Z = np.array(z)
        surf = ax.plot_surface(X, Y, val.T, cmap=cm.jet)
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')
        ax.set_zlabel('$-max_a Q(s,a)$')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

class HashTblEstimator(AbstractEstimator):
    def __init__(self):
        AbstractEstimator.__init__(self)

    def print(self):
        for v in self.estimator.items():
            print(v)
        
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
