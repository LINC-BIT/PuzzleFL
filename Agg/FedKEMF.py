from operator import itemgetter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Agg.AggModel.sixcnn import SixCNN
from Agg.Datasets import ServerDataset, get_server_task_dataset
from Agg.OTFusion import utils, parameters, wasserstein_ensemble
from Agg.AggModel.ensemble import AvgEnsemble
import torch.nn as nn
from copy import deepcopy
def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2
class FedKEMF():
    def __init__(self, agg_sum,datasize=None, sample_num=100, dataname=None, output=100, model = None):
        if model is None:
            self.clients_model = [SixCNN(datasize,output) for i in range(agg_sum)]
        else:
            self.clients_model = [deepcopy(model) for i in range(agg_sum)]
        self.output=output
        self.sample_num = sample_num
        self.agg_args = parameters.get_parameters()
        self.dataname=dataname
        self.device = torch.device("cuda:"+str(self.agg_args.server_gpu))
        if model is None:
            self.server_model = SixCNN(datasize,output)
        else:
            self.server_model = model
        self.loss_fn=nn.MSELoss()
        self.epochs = 10
        self.optimizer_server = torch.optim.Adam(self.server_model.parameters(), lr=0.001)
        # if sample_type is None:
        #     dataset = get_server_dataset(sample_num,name='CIFAR100')
        #     self.dataloader = DataLoader(dataset,batch_size=1, shuffle=False)

    def upload_model(self, all_models):
        for i,client in enumerate(all_models):
            self.clients_model[i].load_state_dict(client)
        self.client_aggmodel = AvgEnsemble(self.clients_model)

    def upload_dataset(self,task):
        dataset = get_server_task_dataset(self.sample_num,name=self.dataname,t=task)
        if self.dataname == 'MiniGC':
            self.dataloader = dataset
        else:
            self.dataloader = DataLoader(dataset,batch_size=20, shuffle=False)

    def update(self,all_models,task):
        self.upload_model(all_models)
        self.upload_dataset(task)
        self.aggregate_model(task)
        return self.server_model.state_dict()

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        # soft_teacher_out = F.softmax(y_pred_teacher / self.temp, dim=1)
        soft_teacher_out = y_pred_teacher
        soft_student_out = F.softmax(y_pred_student / 2, dim=1)

        loss = 0.5 * F.cross_entropy(y_pred_student, y_true)
        loss += 0.5 * self.loss_fn(
            soft_teacher_out, soft_student_out
        )

        return loss
    def aggregate_model(self,t):
        self.server_model.to(self.device)
        self.client_aggmodel.to(self.device)
        if self.dataname == 'MiniGC':
            for epoch in range(self.epochs):
                for images,targets in self.dataloader:
                    images = images.to(self.device)
                    targets = (targets).to(self.device)
                    server_outputs = self.server_model.forward(images)
                    clients_output = self.client_aggmodel.forward(images, -1)
                    loss = self.calculate_kd_loss(server_outputs, clients_output, targets)
                    self.optimizer_server.zero_grad()
                    loss.backward()
                    self.optimizer_server.step()
        else:
            for epoch in range(self.epochs):
                for images,targets in self.dataloader:
                    images = images.to(self.device)
                    targets = (targets - 10*t).to(self.device)
                    if self.output >= 100:
                        server_outputs = self.server_model.forward(images, t)
                        clients_output = self.client_aggmodel.forward(images,t)
                    else:
                        server_outputs = self.server_model.forward(images)
                        clients_output = self.client_aggmodel.forward(images,-1)
                    loss = self.calculate_kd_loss(server_outputs, clients_output, targets)
                    self.optimizer_server.zero_grad()
                    loss.backward()
                    self.optimizer_server.step()
