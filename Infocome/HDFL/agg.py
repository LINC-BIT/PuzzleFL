import torch


class Aggregator():
    def __init__(self, model):
        self.global_model = model
        
    def update(self, models, mix_weights):
        mix_weights = torch.tensor(mix_weights)
        if mix_weights.sum() != 1:
            mix_weights = torch.ones(len(mix_weights))
            mix_weights /= len(mix_weights)
            
        global_state = self.global_model.state_dict()
        
        for key in global_state.keys():
            global_state[key] *= 0
            
        for i, local_model in enumerate(models):
            global_state = self.aggregate_model(global_state, local_model.state_dict(), mix_weights[i])
            
        return global_state
    
    def aggregate_model(self,glob, local, ratio):
        for key in glob.keys():
            if 'num_batches_tracked' not in key and 'feature_net.bn1.running_var' not in key and 'feature_net.bn1.running_mean' not in key:
                glob[key] += local[key]* ratio
            else:
                glob[key] = local[key]
        return glob