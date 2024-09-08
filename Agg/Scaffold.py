import torch
import copy
class Scaffold():
    def __init__(self, model,args):
        self.global_control = {}
        self.global_delta_control = {}
        self.global_delta_y = {}
        self.global_model = copy.deepcopy(model)
        for k, v in self.global_model.named_parameters():
            self.global_control[k] = torch.zeros_like(v.data)
            self.global_delta_control[k] = torch.zeros_like(v.data)
            self.global_delta_y[k] = torch.zeros_like(v.data)
        self.args=args
        # self.nns = []
        # for i in range(self.K):
        #     temp = copy.deepcopy(self.model)
        #     temp.name = self.clients[i]
        #         #     temp.control = copy.deepcopy(self.nn.control)  # ci
        #         #     temp.delta_control = copy.deepcopy(self.nn.delta_control)  # ci
        #         #     temp.delta_y = copy.deepcopy(self.nn.delta_y)
        #         #     self.nns.append(temp)
    def update(self,clients_state_dict):
        self.aggregate_model(clients_state_dict)
        return self.global_model.state_dict()
    def aggregate_model(self, clients_state_dict):
        ratio = 1.0 / len(clients_state_dict)
        # compute
        x = {}
        c = {}
        # init
        for k, v in self.global_model.named_parameters():
            x[k] = torch.zeros_like(v.data)
            c[k] = torch.zeros_like(v.data)

        for client_dict in clients_state_dict:
            for k, v in self.global_model.named_parameters():
                x[k] += client_dict['delta_y'][k] * ratio
                c[k] += client_dict['delta_control'][k] * ratio  # averaging

        # update x and c
        for k, v in self.global_model.named_parameters():
            v.data += x[k].data  # lr=1
            self.global_control[k].data += c[k].data * self.args.frac

