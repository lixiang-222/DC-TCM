# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch
from model import *
import torch.nn as nn
import torch.optim as optim
from train import training, testing
from utils import analyze_drug_statistics
import random
import argparse

# 参数解析
parser = argparse.ArgumentParser(description='中医辨证论治模型训练')
parser.add_argument('--model_name', type=str, default='Ours', help='模型名称 [可选: Ours, SimpleMLP]')
parser.add_argument('--debug', type=bool, default=True, help='调试模式 [True/False]')
parser.add_argument('--embedding_dim', type=int, default=32, help='嵌入维度 [搜索范围: 16, 32,64,128,256]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率 [搜索范围: 0.001,0.005,0.01,0.05,0.1]')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率 [搜索范围: 0.1,0.3,0.5,0.7]')
parser.add_argument('--alpha1', type=float, default=1.0, help='loss1权重 [搜索范围: 0.01,0.05,0.1,0.5,1]')
parser.add_argument('--alpha2', type=float, default=1.0, help='loss2权重 [搜索范围: 0.01,0.05,0.1,0.5,1]')
parser.add_argument('--alpha3', type=float, default=1.0, help='loss3权重 [搜索范围: 0.01,0.05,0.1,0.5,1]')
# parser.add_argument('--alpha4', type=float, default=0.0, help='loss4权重')
parser.add_argument('--alpha5', type=float, default=1.0, help='正交损失权重 [搜索范围: 0.01,0.05,0.1,0.5,1]')
parser.add_argument('--alpha6', type=float, default=0.01, help='对比损失权重 [搜索范围: 0.01,0.05,0.1,0.5,1]')
parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--train', type=bool, default=True, help='是否训练模型')
parser.add_argument('--device', type=str, default='1', help='训练设备')
parser.add_argument('--dataset', type=str, default='Ours', help='数据集 [可选: Ours, tcm_lung]')
args = parser.parse_args()

"""初始化"""
# 设置随机数种子
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # 保证卷积层的结果可重复
    torch.backends.cudnn.benchmark = False  # 关闭加速，确保可重复性

# 检查 GPU 是否可用
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print(f"Using device = cuda:{args.device}")

# 初始化模型、损失函数和优化器
criterions = []
if args.dataset == 'Ours':
    # 加载预处理过的数据
    train_data, test_data = pickle.load(open('processed_data/processed_data_continuous.pkl', 'rb'))
    # train_data, test_data = train_test_split(continuous_data, test_size=0.2, random_state=SEED)
    discrete_data = pickle.load(open('processed_data/processed_data_discrete.pkl', 'rb'))
    X, Y1, Y2, Y3 = discrete_data
    # 假设每个事件中的最大类别数量
    feature_type_num = X.nunique()
    label_num = [Y1.shape[1], Y2.nunique()[0], Y3.shape[1]]
    # 添加损失函数
    criterions.append(nn.BCEWithLogitsLoss())  # 用于 y1 的多标签二分类 y1是辨证，4个症候
    criterions.append(nn.CrossEntropyLoss())  # 用于 y2 的单标签多分类 y2是论治，3个大基础方三选一
    criterions.append(nn.BCEWithLogitsLoss())  # 用于 y3 的多标签二分类 y3是具体处方，323个具体药物
    criterions.append(nn.MSELoss())  # 用于 y4 的多标签回归 y4是具体剂量，323个具体的药物的剂量
elif args.dataset == 'tcm_lung':
    train_data, test_data = pickle.load(open('processed_data/processed_tcm_lung_continuous.pkl', 'rb'))
    discrete_data = pickle.load(open('processed_data/processed_tcm_lung_discrete.pkl', 'rb'))
    X, Y1, Y2, Y3 = discrete_data
    feature_type_num = [2 for _ in range(X.shape[1])]
    label_num = [Y1.shape[1], Y2.shape[1], Y3.shape[1]]
    # 添加损失函数
    criterions.append(nn.BCEWithLogitsLoss())  # 用于 y1 的多标签二分类 y1是辨证，50个症候
    criterions.append(nn.BCEWithLogitsLoss())  # 用于 y2 的多标签二分类 y2是论治 共有61种
    criterions.append(nn.BCEWithLogitsLoss())  # 用于 y3 的多标签二分类 y3是具体处方，379个具体药物
else:
    raise ValueError("Invalid dataset.")

# 添加药物统计
drug_stats = analyze_drug_statistics(Y3)
print("\n药物使用统计:")
print(f"1. 总药物种类: {drug_stats['total_drugs']}种")
print(f"2. 平均每个患者使用药物: {drug_stats['avg_drugs_per_patient']:.2f}种")
print(f"3. 单个患者最大使用药物: {drug_stats['max_drugs_per_patient']}种")

# 定义特征和标签的名称
symptom_name = X.columns.tolist()
syndrome_name = Y1.columns.tolist()
therapeutic_name = ['散结方','肺新方','脾新方']
drug_name = Y3.columns.tolist()

if args.debug:
    args.num_epochs = 3
    train_data = train_data[:10]
    test_data = test_data[:10]

model_path = f"{args.model_name}_{args.embedding_dim}_{args.learning_rate}_{args.dropout}_{args.alpha1}_{args.alpha2}_{args.alpha3}"
if args.model_name == 'SimpleMLP':
    model = SimpleMLP(args.embedding_dim, label_num, feature_type_num, args.dropout, device).to(device)
elif args.model_name == 'Ours':
    model = Ours(args.embedding_dim, label_num, feature_type_num, args.dropout, args.dataset, device).to(device)
elif args.model_name == 'Ours_old':
    model = Ours_old(args.embedding_dim, label_num, feature_type_num, args.dropout, device).to(device)
else:
    raise ValueError("Invalid model name.")

if args.train:
    print("*****Training Phase*****")
    print(f"Model name: {args.model_name}")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 训练
    best_model_pt = training(model, args.model_name, args.num_epochs, train_data, test_data, criterions, optimizer, device, args)
    # 保存模型
    torch.save(best_model_pt, f"saved_model/{model_path}_model.pt")
    print(f"Model saved as {args.model_name}_model.pt\n")

print("*****Inference Phase*****")
best_model_pt = torch.load(f"saved_model/{model_path}_model.pt", weights_only=True)
model.load_state_dict(best_model_pt)
f3_15 = testing(model, args.model_name, test_data, criterions, device, args)
