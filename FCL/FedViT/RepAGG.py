import torch
class FedRepAVG():
    def __init__(self, model):
        self.global_model = model
        
    def update(self,clients_state_dict):
        ratio = 1.0/len(clients_state_dict)
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            if 'last' not in key:
                global_state[key] *= 0
                
        for client_state in clients_state_dict:
            global_state = self.aggregate_model(global_state,client_state,ratio)
        return global_state
    
    def aggregate_model(self,glob, local, ratio):
        for key in glob.keys():
            if 'last' not in key:
                try:
                    glob[key] += local[key]* ratio
                except Exception as e:
                    glob[key] = local[key]
                    
        return glob