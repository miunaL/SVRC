#coding=GBK
from  Factory import *

class SolverContext(object):

    def __init__(self, path="../resource/", model_name = "LR"):
        self.fac = Factory(path, model_name)
    
def run_Svrg(context):
    svrg_params = [0.02, 1]
    svrg = context.fac.get_solver("Svrg", svrg_params)
    svrg_record = svrg.run(100)
    pd.DataFrame([svrg_record.epoch_list, svrg_record.time_list, svrg_record.loss_list]) \
        .to_csv("../resource/svrg_mnist_tmp.csv", header=None)
    
    
if __name__ == "__main__":
    
    context = SolverContext("../resource/", "LR")
    run_Svrg(context)