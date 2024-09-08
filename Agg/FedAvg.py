import torch
class FedAvg():
    def __init__(self, model):
        self.global_model = model
    def update(self,clients_state_dict):
        ratio = 1.0/len(clients_state_dict)
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] *= 0
        for client_state in clients_state_dict:
            global_state = self.aggregate_model(global_state,client_state,ratio)
        return global_state
    def aggregate_model(self,glob, local, ratio):
        for key in glob.keys():
            if 'num_batches_tracked' not in key and 'feature_net.bn1.running_var' not in key and 'feature_net.bn1.running_mean' not in key:
                glob[key] += local[key]* ratio
            else:
                glob[key] = local[key]
        return glob