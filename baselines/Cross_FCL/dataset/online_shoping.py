import torch
import torchtext.data as data
from torchtext.vocab import Vectors
import jieba
import logging
import re
import pandas as pd
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ClientTrain.AggModel.RNN import LSTMMoE
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel

jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
class our_filed(data.Field):
    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>",
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        super(our_filed,self).__init__(sequential=sequential, use_vocab=use_vocab, init_token=init_token,
                 eos_token=eos_token, fix_length=fix_length, dtype=dtype,
                 preprocessing=preprocessing, postprocessing=postprocessing, lower=lower,
                 tokenize=tokenize, tokenizer_language=tokenizer_language, include_lengths=include_lengths,
                 batch_first=batch_first, pad_token=pad_token, unk_token=unk_token,
                 pad_first=pad_first, truncate_first=truncate_first, stop_words=stop_words,
                 is_target=is_target)

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        max_len = max(max_len, 10)
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x)))
            # lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
            lengths.append(len(x))
        if self.include_lengths:
            return (padded, lengths)
        return padded

def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors

def word_cut(text):
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]

class OnlineTask():
    def __init__(self):
        pass

    def del_tensor_ele(arr, index):
        arr1 = arr[0:index]
        arr2 = arr[index + 1:]
        return torch.cat((arr1, arr2), dim=0)

    def process_cl_dataset(self):
        df = pd.read_csv('/data/lpyx/data/OnlineShopping/online_shopping_10_cats.csv')
        cats = df['cat'].unique()
        datas = []
        for c in cats:
            datas.append({'cat':c,'data':df[df['cat'] == c]})
        for d in datas:
            d['data'].to_csv('/data/lpyx/data/OnlineShopping/cl/'+d['cat']+'.csv',index=None)

    def get_multi_dataset(self, text_field, label_field):
        vectors = load_word_vectors('data.vector', '/data/lpyx/data/OnlineShopping/vector/')
        text_field.tokenize = word_cut
        cl_dir = ['/data/lpyx/data/OnlineShopping/cl/' + d for d in os.listdir('/data/lpyx/data/OnlineShopping/cl/')]
        cl_datasets = []
        for i in range(len(cl_dir)):
            cur_text_field = copy.deepcopy(text_field)
            cur_label_field = copy.deepcopy(label_field)
            cur_dataset = data.TabularDataset(path=cl_dir[i],
                                              format='csv',
                                              fields=[('cat', None), ('label', cur_label_field),
                                                      ('text', cur_text_field)])
            cl_datasets.append({'text_field': cur_text_field, 'label_field': cur_label_field, 'dataset': cur_dataset})

        cl_train_datasets = []
        cl_test_datasets = []
        for cl_data in cl_datasets:
            train, dev = cl_data['dataset'].split(
                split_ratio=0.8
            )
            cl_data['text_field'].build_vocab(train, dev, vectors=vectors)

            cl_data['label_field'].build_vocab(train, dev)
            train_iter, dev_iter = data.Iterator.splits(
                (train, dev),
                batch_sizes=(len(train), len(dev)),
                sort_key=lambda x: len(x.text))
            cl_train_datasets.append(OnlineDataset(cl_data['text_field'], cl_data['label_field'], train_iter))
            cl_test_datasets.append(OnlineDataset(cl_data['text_field'], cl_data['label_field'], dev_iter))
        return cl_train_datasets, cl_test_datasets

    def get_single_dataset(path, text_field, label_field):
        text_field.tokenize = word_cut
        online_data = data.TabularDataset(path='/data/lpyx/data/OnlineShopping/online_shopping_10_cats.csv',
                                          format='csv',
                                          fields=[('cat', None), ('label', label_field), ('text', text_field)])
        train, dev = online_data.split(
            split_ratio=0.8
        )
        vectors = load_word_vectors('data.vector', '/data/lpyx/data/OnlineShopping/vector/')
        text_field.build_vocab(train, dev, vectors=vectors)

        label_field.build_vocab(train, dev)
        return train, dev

    def getTaskDataSet(self, choice = 'multi'):
        text_field = our_filed(lower=True,include_lengths=True)
        label_field = data.Field(sequential=False)
        if choice == 'multi':
            return self.get_multi_dataset(text_field,label_field)
        else:
            return self.get_single_dataset(text_field, label_field)

class OnlineDataset(Dataset):
    def __init__(self, text_field,label_field, data):
        self.text_field = text_field
        self.label_field = label_field
        vocabulary_size = len(self.text_field.vocab)
        embedding_dimension = self.text_field.vocab.vectors.size()[-1]
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.embedding = self.embedding.from_pretrained(self.text_field.vocab.vectors, freeze=True)
        for batch in data:
            self.data= batch

        self.texts,self.length = self.data.text
        self.texts = self.texts.t_()
        self.texts = self.embedding(self.texts)
        # self.texts = self.texts.unsqueeze(1)
        self.labels = self.data.label
        self.labels.data.sub_(1)
        re_index = torch.nonzero(self.length).squeeze()
        self.texts = torch.index_select(self.texts,0,re_index)
        self.labels = torch.index_select(self.labels,0,re_index)
        self.length = torch.index_select(self.length,0,re_index)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        feature, target,length = self.texts[item], self.labels[item], self.length[item]
        return feature,target,length



class Noemb_TextCNN(nn.Module):
    def __init__(self):
        super(Noemb_TextCNN, self).__init__()

        embedding_dimension = 100
        class_num = 4
        chanel_num = 1
        filter_num = 1
        filter_sizes =[3,4,5]
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension),bias=False) for size in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num, bias=False)

    def forward(self, x):
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for feature, target,length in data_iter:
        feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def train(data_iter, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    i = 0
    for feature, target,length in data_iter:
        model.train()
        feature, target = feature.cuda(), target.cuda()
        optimizer.zero_grad()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.embedding_dim = 100
        self.hidden_dim = 128
        self.output_dim = 4
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.fc_emb = nn.Linear(self.embedding_dim , 768)
        self.fc = nn.Linear(self.embedding_dim, self.hidden_dim )
        self.output_layer = nn.Linear(self.hidden_dim , self.output_dim)

    def forward(self, embedded):
        # 如果已经嵌入了样本，则直接将嵌入后的样本作为输入
        # embedded 的维度为 [batch_size, sequence_length, embedding_dim]
        # 这里假设已经经过了某种嵌入操作
        out = self.fc_emb(embedded)
        bert_output = self.bert(inputs_embeds=out)
        pooled_output = bert_output.pooler_output
        hidden = self.fc(pooled_output)
        output = self.output_layer(hidden)
        return output
if __name__ == "__main__":
    # model = LSTMMoE().cuda()
    model = Classifier().cuda()
    # model.load_state_dict(torch.load('/data/lpyx/FedAgg/Agg/test_text_model/center1.pt'))
    online = OnlineTask()
    trains,tests = online.getTaskDataSet()
    for train_dataset,test_dataset in zip(trains,tests):
        train_iter = DataLoader(train_dataset, batch_size=128, shuffle=True)
        dev_iter = DataLoader(test_dataset,batch_size=64, shuffle=True)
        # for i in range(20):
        for i in range(10):
            train(train_iter, model)
            eval(dev_iter, model)
            # torch.save(model.state_dict(), '/data/lpyx/FedAgg/Agg/test_text_model/center' + str(i) + '.pt')
        break


