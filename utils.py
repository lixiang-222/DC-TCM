from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, ndcg_score, confusion_matrix, mean_squared_error, mean_absolute_error
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# def classification_evaluate_metrics(y_true, y_pred, type):
#     if type == 'multiclass':
#         # 计算多分类任务的 F1 分数，使用宏平均
#         f1 = f1_score(y_true, y_pred, average="macro")

#         # 计算准确率：正确预测的样本数与总样本数之比
#         accuracy = accuracy_score(y_true, y_pred)

#     else:
#         # # 使用 average="samples" 表示逐样本计算 Jaccard，相当于每行的交并比
#         # jaccard = jaccard_score(y_true, y_pred, average="samples")
#         # # print(f"\nAverage Jaccard Score: {jaccard:.4f}")

#         # 使用 average="samples" 表示逐样本计算 F1-Score，并求平均值
#         f1 = f1_score(y_true, y_pred, average="samples")

#         label_accuracies = []
#         num_labels = len(y_true[0])  # 标签的数量（假设每个样本有相同数量的标签）

#         for i in range(num_labels):
#             # 提取每个标签的真实值和预测值
#             y_true_label = [y[i] for y in y_true]
#             y_pred_label = [y[i] for y in y_pred]

#             # 计算该标签的准确率
#             acc = accuracy_score(y_true_label, y_pred_label)
#             label_accuracies.append(acc)

#         # 计算所有标签准确率的平均值
#         accuracy = np.mean(label_accuracies)

#     return accuracy, f1

def classification_evaluate_metrics(y_true, y_pred, type):
    if type == 'multiclass':
        # 计算多分类指标
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 初始化指标存储
        specificity_list = []
        sensitivity_list = []
        
        # 遍历每个类别
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)
            
            # 计算每个类别的specificity和sensitivity
            specificity = tn / (tn + fp + 1e-7)
            sensitivity = tp / (tp + fn + 1e-7)
            
            specificity_list.append(specificity)
            sensitivity_list.append(sensitivity)
        
        # 计算宏平均
        specificity = np.mean(specificity_list)
        sensitivity = np.mean(sensitivity_list)
        
    else:
        # 多标签指标计算
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="samples")
        
        # 初始化指标存储
        specificity_list = []
        sensitivity_list = []
        
        # 遍历每个标签
        num_labels = y_true.shape[1]
        for i in range(num_labels):
            # 提取当前标签的预测和真实值
            true = y_true[:, i]
            pred = y_pred[:, i]
            
            # 计算混淆矩阵元素
            tn = np.sum((true == 0) & (pred == 0))
            fp = np.sum((true == 0) & (pred == 1))
            tp = np.sum((true == 1) & (pred == 1))
            fn = np.sum((true == 1) & (pred == 0))
            
            # 计算当前标签的指标
            specificity = tn / (tn + fp + 1e-7)
            sensitivity = tp / (tp + fn + 1e-7)
            
            specificity_list.append(specificity)
            sensitivity_list.append(sensitivity)
        
        # 计算平均值
        specificity = np.mean(specificity_list)
        sensitivity = np.mean(sensitivity_list)
    
    return accuracy, f1, specificity, sensitivity

def evaluate_y3_at_k(y_true, y_pred_scores, k_list=[5, 10, 15, 20]):
    """多标签排名指标评估（不含NDCG）"""
    # 转换为numpy数组确保后续操作安全
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred_scores = np.asarray(y_pred_scores, dtype=np.float32)
    
    # 初始化指标存储字典
    metrics = {
        'precision@k': {k: [] for k in k_list},
        'recall@k': {k: [] for k in k_list},
        'f1@k': {k: [] for k in k_list}
    }
    
    # 遍历每个样本单独处理
    for i in range(len(y_true)):
        # 获取当前样本数据
        scores = y_pred_scores[i]
        true_labels = y_true[i]
        
        # 生成所有k值的预测
        sorted_idx = np.argsort(scores)[::-1]  # 按分数降序排列的索引
        
        for k in k_list:
            # 生成top-k预测标签（0/1）
            pred_labels = np.zeros_like(scores)
            pred_labels[sorted_idx[:k]] = 1
            
            # 计算当前样本的指标
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            
            # 存储结果
            metrics['precision@k'][k].append(precision)
            metrics['recall@k'][k].append(recall)
            metrics['f1@k'][k].append(f1)
    
    # 计算平均指标并转换结构
    final_metrics = {
        'precision@k': [],
        'recall@k': [],
        'f1@k': []
    }
    
    for k in sorted(k_list):
        final_metrics['precision@k'].append(np.mean(metrics['precision@k'][k]))
        final_metrics['recall@k'].append(np.mean(metrics['recall@k'][k]))
        final_metrics['f1@k'].append(np.mean(metrics['f1@k'][k]))
    
    return final_metrics

def regression_evaluate_metrics(y_true, y_pred):
    # 1. 计算 Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    # print(f"\nMean Squared Error (MSE): {mse:.4f}")

    # 2. 计算 Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    # print(f"\nMean Absolute Error (MAE): {mae:.4f}")

    # 3. 计算 Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")
    return mse, mae, rmse


def plot_losses(train_losses, val_losses, model_name):
    png_path = f'./results/{model_name}_loss.png'
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    # 设置纵坐标范围
    plt.autoscale(True)
    # 保存绘图
    plt.savefig(png_path)
    plt.close()


def round_5(output4):
    # 1. 先计算除以5后的值
    scaled_doses = output4 / 5

    # 2. 对不同范围的值进行不同的处理：
    #    - ≤0 → 0
    #    - 0 < x < 1 → 1 (即0-5 → 5)
    #    - ≥1 → 四舍五入到最近的整数再乘以5
    rounded_scaled = torch.where(
        scaled_doses <= 0,
        torch.zeros_like(scaled_doses),  # ≤0 → 0
        torch.where(
            scaled_doses < 1,
            torch.ones_like(scaled_doses),  # 0 < x < 1 → 1 (即0-5 → 5)
            torch.round(scaled_doses)  # ≥1 → 四舍五入
        )
    )

    # 3. 最终剂量 = 处理后的值 ×5
    rounded_doses = rounded_scaled * 5
    return rounded_doses


def analyze_drug_statistics(Y3):
    """分析药物使用统计数据
    
    参数:
        Y3: DataFrame, 包含所有药物使用情况，列为药物名称，行为患者
    
    返回:
        dict: 包含药物统计信息的字典
    """
    # 1. 药物总数
    total_drugs = len(Y3.columns)
    
    # 2. 计算每个患者的药物使用数量
    drug_counts = (Y3 != 0).sum(axis=1)
    
    # 3. 平均使用药物数
    avg_drugs = drug_counts.mean()
    
    # 4. 最大使用药物数
    max_drugs = drug_counts.max()
    
    return {
        'total_drugs': total_drugs,
        'avg_drugs_per_patient': avg_drugs,
        'max_drugs_per_patient': max_drugs
    }