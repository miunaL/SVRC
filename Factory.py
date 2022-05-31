#coding=GBK

from numpy import *
from loadData import *
from LRSoftmax import *
from LR import *
import svrg_ as svrg


from Logger import *

class Factory(object):
    def __init__(self, path="../resource/", model_name = "LR"):
        self.path = path
        self.model = self.get_model(model_name)
        self.model_name = model_name

    def get_model(self, model_name):
        if model_name == "LR" :
            data_x, data_y = getMnist49(path=self.path)
            return LR(data_x, data_y, 1e-4)
        elif model_name == "LRSoftmax":
            data_x, data_y = getMnistWithNumber(path=self.path, number = array([0,1,2,3,4,5,6,7,8,9]))
            # print("data_x mean",data_x.mean(axis=1))
            return LRSoftmax(data_x, data_y, 1e-4)
        else:
            pass

    def get_solver(self, solver_name, params):

        if solver_name == "Svrg":
            return svrg.Svrg(self.model, params[0], params[1])
        else:
            Logger.log(solver_name+" not implement")
            pass

    def get_solver_with_rand_params(self, solver_name):
        w_scale_lists = np.power(0.1, np.arange(3) + 3)
        w_scale = asscalar(random.choice(w_scale_lists, 1))
        if solver_name == 'Svrg':
            # parameters in paper
            if self.model_name == "LRSoftmax":
                # step_size = asscalar(random.choice((arange(22) + 8) * 0.05, 1))
                return self.get_solver(solver_name, [0.025, 2])
            else:
                step_size = asscalar(random.choice((arange(26) + 4) * 0.05, 1))
                s1 = asscalar(random.choice(np.arange(1) + 1, 1))
                return self.get_solver(solver_name, [step_size, s1])
        else:
            pass