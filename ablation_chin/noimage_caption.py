import json
import os
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, BertModel

# Config
config = {
    "learning_rate": 1.5e-5,
    "batch_size": 10,
    "num_epochs": 5,
    "model_save_path": "noimage_cap_chin.pth",
    "early_stopping_rounds": 5
}

from transformers import BertTokenizer

class RumorDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')  # 使用BertTokenizer
        self.samples = self._load_dataset()

    def _load_dataset(self):
        samples = []
        for label in ['fake', 'real']:
            label_dir = os.path.join(self.root_dir, 'weibo', label) # 修改路径
            for weibo_id in os.listdir(label_dir):
                weibo_path = os.path.join(label_dir, weibo_id)
                if os.path.isdir(weibo_path):
                    with open(os.path.join(weibo_path, 'content.json'), 'r', encoding='utf-8') as file:
                        content = json.load(file)
                    text = content['text'] # 假设content.json中有文本字段

                    # 找到.jpg文件
                    img_files = [f for f in os.listdir(weibo_path) if f.endswith('.jpg')]
                    if img_files:
                        img_path = os.path.join(weibo_path, img_files[0])
                        samples.append((text, img_path, label))  # 不再包含图片描述
        return samples

    def __getitem__(self, idx):
        text, img_path, label = self.samples[idx]

        # 仅使用文本进行编码
        encoded_text = self.tokenizer.encode_plus(
            text, add_special_tokens=True, return_tensors='pt',
            max_length=512, truncation=True, padding='max_length'
        )
        input_ids = encoded_text['input_ids'].squeeze(0)
        attention_mask = encoded_text['attention_mask'].squeeze(0)
        label = 1 if label == 'real' else 0

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return input_ids, attention_mask, image, label

    def __len__(self):
        return len(self.samples)

# 其他代码部分不变



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# dataset = RumorDetectionDataset(root_dir='total_dataset', transform=transform)
# data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

#model
from torch import nn
from transformers import RobertaModel
from torchvision.models import resnet50
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch

# 检查CUDA是否可用并设置默认设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RumorDetectionModel(nn.Module):
    def __init__(self):
        super(RumorDetectionModel, self).__init__()
        self.text_model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

        # 分类器
        self.classifier = nn.Linear(self.text_model.config.hidden_size, 2)  # 只使用文本特征
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, images):
        # 提取文本特征
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        text_features = text_features[:, 0, :]  # 取CLS标记的特征

        # 分类
        text_features = self.dropout(text_features)
        logits = self.classifier(text_features)
        return logits

model = RumorDetectionModel().to(device)

#train
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, images, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask, images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 扩展评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    bi_precision, bi_recall, bi_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return {
        "accuracy": accuracy,
        "bi_precision": bi_precision, "bi_recall": bi_recall, "bi_f1": bi_f1,
        "micro_precision": micro_precision, "micro_recall": micro_recall, "micro_f1": micro_f1,
        "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1,
        "weighted_precision": weighted_precision, "weighted_recall": weighted_recall, "weighted_f1": weighted_f1
    }

def train_model(model, train_loader, val_loader, epochs, fold_number):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = CrossEntropyLoss()
    best_f1 = 0
    best_model = None
    no_improvement = 0
    early_stopping_rounds = config['early_stopping_rounds']

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, images, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 每五十个 batch 展示一次数据
            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"Fold {fold_number}, Epoch {epoch}, Batch {i+1}, Avg Loss: {avg_loss}")

            # 每一百个 batch 或在最后一个 batch 计算验证集指标
            if (i + 1) % 200 == 0 or i == len(train_loader) - 1:
                val_metrics = evaluate_model(model, val_loader)
                print(f"Fold {fold_number}, Epoch {epoch}, Batch {i+1}, Validation Metrics: {val_metrics}")
                if val_metrics['bi_f1'] > best_f1:
                    best_f1 = val_metrics['bi_f1']
                    best_model = model.state_dict()
                    fold_model_save_path = config['model_save_path'].replace('.pth', f'_fold_{fold_number}.pth')
                    torch.save(best_model, fold_model_save_path)
                    no_improvement = 0
                else:
                    no_improvement += 1

        if no_improvement >= early_stopping_rounds:
            print(f"Early stopping triggered after {epoch + 1} epochs in Fold {fold_number}")
            break

import pandas as pd
def test_model(model, test_loader, experiment_name, fold_number, csv_path='noimage_cap_results_chin.csv'):
    # 加载指定 fold 的最佳模型
    fold_model_save_path = config['model_save_path'].replace('.pth', f'_fold_{fold_number}.pth')
    model.load_state_dict(torch.load(fold_model_save_path))

    # 获取测试指标
    test_metrics = evaluate_model(model, test_loader)

    # 添加实验名称到测试指标中
    test_metrics['experiment_name'] = experiment_name

    # 将测试指标转换为DataFrame
    test_metrics_df = pd.DataFrame([test_metrics])

    # 检查CSV文件是否存在
    try:
        with open(csv_path, 'x') as f:
            test_metrics_df.to_csv(f, index=False)
    except FileExistsError:
        with open(csv_path, 'a') as f:
            test_metrics_df.to_csv(f, header=False, index=False)

    # 打印测试指标
    print("Test Metrics:", test_metrics)
    return test_metrics



#main
#main
from sklearn.model_selection import KFold

if __name__ == "__main__":
    # 原始数据集
    dataset = RumorDetectionDataset(root_dir='E:\\PycharmProjects\\pmccRes\\total_CLIPCAP', transform=transform)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 用于存储所有fold的结果
    all_fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset.samples)):
        print(f"Fold {fold+1}")

        # 为每个 fold 重新初始化模型
        model = RumorDetectionModel().to(device)

        # 分割数据为当前fold的训练集和验证集
        train_samples = [dataset.samples[i] for i in train_ids]
        val_samples = [dataset.samples[i] for i in val_ids]

        # 训练集和验证集的DataLoader
        train_dataset = RumorDetectionDataset(root_dir='E:\\PycharmProjects\\pmccRes\\total_CLIPCAP', transform=transform)
        train_dataset.samples = train_samples
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

        val_dataset = RumorDetectionDataset(root_dir='E:\\PycharmProjects\\pmccRes\\total_CLIPCAP', transform=transform)
        val_dataset.samples = val_samples
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

        # 训练模型
        train_model(model, train_loader, val_loader, config['num_epochs'], fold+1)

        # 测试模型
        fold_result = test_model(model, val_loader, experiment_name=f'fold_{fold+1}', fold_number=fold+1, csv_path='noimage_cap_results_chin.csv')
        all_fold_results.append(fold_result)

    # 计算所有fold的平均指标并保存到CSV
    # 确保仅对数值类型的结果计算平均值
    numeric_results = [{k: v for k, v in result.items() if isinstance(v, (int, float))} for result in all_fold_results]
    avg_metrics = {key: np.mean([result[key] for result in numeric_results]) for key in numeric_results[0]}
    avg_metrics_df = pd.DataFrame([avg_metrics])
    avg_metrics_df.to_csv('noimage_avg_cap_chin.csv', index=False)




