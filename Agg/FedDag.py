from operator import itemgetter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Agg.AggModel.sixcnn import SixCNN
from Agg.Datasets import ServerDataset, get_server_dataset
from Agg.OTFusion import utils, parameters, wasserstein_ensemble
from random import sample
import copy
def MultiClassCrossEntropy(logits, labels, T=1):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    label = torch.softmax(labels / T, dim=1)
        # print('outputs: ', outputs)
        # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * label, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)

    # print('OUT: ', outputs)
    return outputs
class FedDag():
    def __init__(self, center_nums, candidate_nums, datasize=None, sample_num=5, dataname=None, output=100, model=None):
        if model is not None:
            self.center_models = [{'client': 1, 'model': copy.deepcopy(model)} for i in
                                  range(center_nums)]
            self.candidate_models = [copy.deepcopy(model) for i in range(candidate_nums)]
        else:
            self.center_models = [{'client':1,'model':SixCNN(datasize,outputsize=output)} for i in range(center_nums)]
            self.candidate_models = [SixCNN(datasize,output) for i in range(candidate_nums)]
        self.select_his_num = candidate_nums - center_nums
        self.output=output
        self.sample_num = sample_num
        self.agg_args = parameters.get_parameters()
        self.dataname=dataname
        self.device = torch.device("cuda:"+str(self.agg_args.server_gpu))
        self.k = 4
        self.his_knowledge = []
        # if sample_type is None:
        #     dataset = get_server_dataset(sample_num,name='CIFAR100')
        #     self.dataloader = DataLoader(dataset,batch_size=1, shuffle=False)

    def decode_candidate_model(self, model, encode_dict):
        encode_flag = True
        for k in encode_dict.keys():
            if len(encode_dict[k]) == 1:
                encode_flag = False
            break
        if encode_flag:
            print('decode')
            for name, parameter in model.named_parameters():
                cur_all_data = torch.flatten(parameter.data)
                cur_position = encode_dict[name]['position']
                big_weight = encode_dict[name]['weight']
                new_weight = torch.zeros(cur_all_data.shape)
                new_weight.scatter_(0, cur_position, big_weight)
                parameter.data = new_weight.view(parameter.data.shape)
        else:
            model.load_state_dict(encode_dict)
        return model

    def upload_model(self, all_models):
        center_index = 0
        candidate_index = 0
        for clients in all_models:
            client_id = clients['client']
            model_dict = clients['models']
            self.center_models[center_index]['model'].load_state_dict(model_dict)
            self.center_models[center_index]['model']= self.center_models[center_index]['model'].to(self.device)
            self.center_models[center_index]['client'] = client_id
            center_index += 1
            self.candidate_models[candidate_index].load_state_dict(model_dict)
            self.candidate_models[candidate_index]= self.candidate_models[candidate_index].to(self.device)
            candidate_index+=1
        if self.select_his_num >= len(self.his_knowledge):
            for know in  self.his_knowledge:
                self.candidate_models[candidate_index] = self.decode_candidate_model(self.candidate_models[candidate_index], know)
                self.candidate_models[candidate_index] = self.candidate_models[candidate_index].to(self.device)
                candidate_index += 1
        else:
            select_his_know = sample(self.his_knowledge, self.select_his_num)
            for know in select_his_know:
                self.candidate_models[candidate_index] = self.decode_candidate_model(
                    self.candidate_models[candidate_index], know)
                self.candidate_models[candidate_index] = self.candidate_models[candidate_index].to(self.device)
                candidate_index += 1
        return candidate_index

    def upload_dataset(self,task):
        dataset = get_server_dataset(self.sample_num,name=self.dataname,t=task)
        if self.dataname =='MiniGC':
            self.dataloader = dataset
        else:
            self.dataloader = DataLoader(dataset,batch_size=1, shuffle=False)
    def select_model(self,task,model_num):
        agg_num = self.k
        if model_num < self.k:
            agg_num = model_num

        ## 中心集选择候选集
        select_models=[]
        for model_dict in self.center_models:
            cur_center = model_dict['model']
            cur_center.eval()
            similarity=[]
            for number,cur_candidate in enumerate(self.candidate_models):
                if number < model_num:
                    loss=0
                    cur_candidate.eval()
                    for x in self.dataloader:
                        cen_y = cur_center(x.to(self.device),task)
                        can_y = cur_candidate(x.to(self.device), task)
                        loss += MultiClassCrossEntropy(cen_y,can_y)
                    similarity.append({'number':number,'sim':loss})
            similarity = sorted(similarity, key=itemgetter('sim'), reverse=False)

            model_dict['similarity']=similarity[0:agg_num]
            select_models.append(model_dict)
        return select_models

    def update(self,all_models,task):
        agg_num = self.upload_model(all_models)
        self.upload_dataset(task)
        select_models = self.select_model(task,agg_num)
        agg_models = self.aggregate_model(select_models)
        return agg_models

    def aggregate_model(self,select_models):
        agg_models=[]
        for select_model in select_models:
            center_model = select_model['model']
            client_id = select_model['client']
            candidate_models = [self.candidate_models[i['number']] for i in select_model['similarity']]
            # center_activation = utils.get_model_activations(self.agg_args, [center_model], dataloader=self.dataloader)
            candidate_activations  = utils.get_model_activations(self.agg_args, candidate_models, dataloader=self.dataloader)
            geometric_model = center_model
            for candidate_model, candidate_activation in zip(candidate_models,candidate_activations):
                geometric_activation = utils.get_model_activations(self.agg_args, [geometric_model], dataloader=self.dataloader)
                fusion_models = [candidate_model , geometric_model]
                fusion_activations = {0: candidate_activations[candidate_activation],1:geometric_activation[0]}
                _, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(self.agg_args, fusion_models,
                                                                                           self.dataloader,
                                                                                           self.dataloader,
                                                                                           fusion_activations,output=self.output)

            agg_models.append({'client':client_id,'model':geometric_model.state_dict()})



            # if type(center_activation) == type([]):
            #     all_activations = [[candidate_activation, center_activation] for candidate_activation in candidate_activations]
            # else:
            #     all_activations = [{0:candidate_activations[candidate_activation], 1:center_activation[0]} for candidate_activation in candidate_activations]
            # all_models = [[candidate_model, center_model] for candidate_model in candidate_models]
            # for activations, models in zip(all_activations,all_models):
            #     _, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(self.agg_args, models,
            #                                                                                        self.dataloader,
            #                                                                                        self.dataloader,
            #                                                                                        activations)
            #     agg_models.append(geometric_model)
        return agg_models

    def add_history(self,client_dict):
        self.his_knowledge.append(client_dict)





