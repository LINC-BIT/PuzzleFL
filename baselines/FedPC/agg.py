import torch


class Aggregator():
    def __init__(self, model):
        self.global_model = model
        
    def update_fedhp(self, models):
        # miu 是当前Round中，客户端邻域的最大值，比如每个client有2个邻居进行聚合，则miu为2
        
        cur_client_model = models[0]    
        self.global_model.load_state_dict(cur_client_model.state_dict())
        global_state = self.global_model.state_dict()
        
        miu = 1/len(models)
        for i, local_model in enumerate(models):
            if i > 0:
                global_state = self.aggregate_model(global_state, local_model.state_dict(), miu)
            
        return global_state
    
    def update(self, models):
        mix_weights = torch.ones(len(models))
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