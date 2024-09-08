
from Agg.serverbase import Server
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
from Agg.AggModel.generator import Generator
MIN_SAMPLES_PER_LABEL=1
def get_dataset_name(dataset):
    dataset=dataset.lower()
    passed_dataset=dataset.lower()
    if 'celeb' in dataset:
        passed_dataset='celeb'
    elif 'emnist' in dataset:
        passed_dataset='emnist'
    elif 'mnist' in dataset:
        passed_dataset='mnist'
    elif 'cifar' in dataset:
        passed_dataset = 'cifar'

    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset
def create_generative_model(dataset, algorithm='', model='cnn', embedding=False):
    passed_dataset=get_dataset_name(dataset)
    assert any([alg in algorithm for alg in ['FedGen', 'FedGen']])
    if 'FedGen' in algorithm:
        # temporary roundabout to figure out the sensitivity of the generator network & sampling size
        if 'cnn' in algorithm:
            gen_model = algorithm.split('-')[1]
            passed_dataset+='-' + gen_model
        elif '-gen' in algorithm: # we use more lightweight network for sensitivity analysis
            passed_dataset += '-cnn1'
    return Generator(passed_dataset, model=model, embedding=embedding, latent_layer_idx=-1)
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
class FedGen(Server):
    def __init__(self, args, model):
        super().__init__(args, model)

        # Initialize data for all users
        # data contains: clients, groups, train_data, test_data, proxy_data
        self.total_test_samples = 0
        self.batch_size = 32

        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.student_model = copy.deepcopy(self.model)
        self.generative_model = create_generative_model(args.dataset, 'FedGen', 'cnn', args.embedding)
        # if not args.train:
        #     print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
        #     print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)

        self.clients_models = [copy.deepcopy(self.model) for i in range(args.num_users)]
        self.device = torch.device('cpu')


        #### creating users ####
        # self.users = []
        # for i in range(total_users):
        #     id, train_data, test_data, label_info =read_user_data(i, data, dataset=args.dataset, count_labels=True)
        #     self.total_train_samples+=len(train_data)
        #     self.total_test_samples += len(test_data)
        #     id, train, test=read_user_data(i, data, dataset=args.dataset)
        #     user=UserpFedGen(
        #         args, id, model, self.generative_model,
        #         train_data, test_data,
        #         self.available_labels, self.latent_layer_idx, label_info,
        #         use_adam=self.use_adam)
        #     self.users.append(user)
        # print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        # print("Data from {} users in total.".format(total_users))
        # print("Finished creating FedAvg server.")
    def send_parameters(self):
        return self.model.state_dict()

    def aggregate(self,t, clients_parameters,clients_label_counts):
        for i,client_parameter in enumerate(clients_parameters):
            self.clients_models[i].load_state_dict(client_parameter)
        self.selected_number = len(clients_parameters)
        self.clients_label_counts = clients_label_counts
        self.generative_model.to(self.device)
        self.train_generator(
            t,
            self.batch_size,
            epoches=self.ensemble_epochs // self.n_teacher_iters,
            latent_layer_idx=self.latent_layer_idx,
            verbose=True
        )
        self.aggregate_parameters()
        return self.send_parameters()
        # curr_timestamp=time.time()  # log  server-agg end time
        # agg_time = curr_timestamp - self.timestamp
        # self.metrics['server_agg_time'].append(agg_time)
        # if glob_iter  > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
        #     self.visualize_images(self.generative_model, glob_iter, repeats=10)

    def add_parameters(self, user_model, ratio):
        for server_param, user_param in zip(self.model.parameters(), user_model.parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        for i,user_model in enumerate(self.clients_models):
            self.add_parameters(user_model, 1 / self.selected_number)
            if i == self.selected_number:
                break
    def train_generator(self, t,batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        #self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        def update_generator_(t, n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y=np.random.choice(self.qualified_labels, batch_size)
                y_input=torch.LongTensor(y)
                ## feed to generator
                gen_result=self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss=0
                teacher_logit=0
                for user_idx in range(self.selected_number):
                    self.clients_models[user_idx].eval()
                    weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                    expand_weight=np.tile(weight, (1, self.unique_labels))
                    offset1, offset2 = compute_offsets(t, 10)
                    user_result_given_gen=self.clients_models[user_idx](gen_output,t, start_layer_idx=latent_layer_idx)[:, offset1:offset2]
                    user_output_logp_=F.log_softmax(user_result_given_gen, dim=1)
                    teacher_loss_=torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32))
                    teacher_loss+=teacher_loss_
                    teacher_logit+=user_result_given_gen * torch.tensor(expand_weight, dtype=torch.float32)

                ######### get student loss ############
                offset1, offset2 = compute_offsets(t, 10)
                student_output=student_model(gen_output, start_layer_idx=latent_layer_idx)[:, offset1:offset2]
                student_loss=F.kl_div(F.log_softmax(student_output, dim=1), F.softmax(teacher_logit, dim=1))
                if self.ensemble_beta > 0:
                    loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(t,self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()


    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for label_counts in self.clients_label_counts:
                weights.append(label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y=self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input=torch.tensor(y)
        generator.eval()
        images=generator(y_input, latent=False)['output'] # 0,1,..,K, 0,1,...,K
        images=images.view(repeats, -1, *images.shape[1:])
        images=images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))
