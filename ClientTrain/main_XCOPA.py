import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import json
from typing import List
from sklearn.metrics import accuracy_score
from datasets import load_dataset


# 设置随机数种子
torch.manual_seed(42)

device = torch.device('cuda:0')
# 加载XCOPA数据集
class XCOPADataset(Dataset):
    def __init__(self, path: str, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        self.sentences = []
        self.labels = []
        file = open(path, 'r', encoding='utf-8')
        data = []
        for line in file.readlines():
            dic = json.loads(line)
            data.append(dic)

        for item in data:
            if item['label'] == 0:
                self.sentences.append(item["premise"] + " " + item["choice1"])
                self.labels.append(0)
                self.sentences.append(item["premise"] + " " + item["choice2"])
                self.labels.append(1)
            else:
                self.sentences.append(item["premise"] + " " + item["choice1"])
                self.labels.append(1)
                self.sentences.append(item["premise"] + " " + item["choice2"])
                self.labels.append(0)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.labels[idx]
        encoded_text = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.sentences)

# 定义模型
class XCOPA_Model(nn.Module):
    def __init__(self, num_labels):
        super(XCOPA_Model, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        return output.logits

# 定义训练函数
def train(model, optimizer, criterion, train_dataloader):
    model.train()
    model = model.to(device)
    train_loss = 0
    train_acc = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[2].to(device)
        output = model(input_ids, attention_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        train_acc += accuracy_score(preds.cpu(), label.cpu())
    return train_loss / len(train_dataloader), train_acc / len(train_dataloader)

# 定义测试函数
def evaluate(model, criterion, test_dataloader):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            label = batch[2].to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output, label)
            test_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            test_acc += accuracy_score(preds.cpu(), label.cpu())
    return test_loss / len(test_dataloader), test_acc / len(test_dataloader)

# 加载XCOPA数据集
# xcopa_dataset = load_dataset('/data/lpyx/data/XCOPA/xcopa-master/data-gmt/zh/')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = XCOPADataset("/data/lpyx/data/XCOPA/xcopa-master/data-gmt/zh/test.zh.jsonl", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = XCOPADataset("/data/lpyx/data/XCOPA/xcopa-master/data-gmt/zh/val.zh.jsonl", tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# 定义模型和
batch_size = 32
learning_rate = 0.001
num_epochs = 10
model = XCOPA_Model(2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for e in range(5):
    train(model,optimizer,criterion,train_loader)
    l,a = evaluate(model,criterion,test_loader)
    print(a)