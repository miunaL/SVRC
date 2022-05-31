#coding=GBK

import numpy as np
from Solver import *
from model import *
from Logger import *

# svrg
class Svrg(Solver):
    def __init__(self, model, step_size, m):
        Solver.__init__(self, model, step_size)
        self.m = m
        self.d = self.model.data_x.shape[0]
        self.n = self.model.data_x.shape[1]
        self.k = self.model.data_y.shape[0]
        self.name = "Svrg"

    def get_params(self):
        params = {"step_size": self.step_size, "m": self.m}
        return params

