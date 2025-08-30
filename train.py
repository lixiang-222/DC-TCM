from tqdm import tqdm  # 进度条
from utils import *
import numpy as np
import torch
import datetime
import csv
import os
import datetime
import random

def evaluation(model, model_name, test_data, criterions, device, args):
    """评估模型在数据集上的表现"""
    model.eval()
    all_preds1 = []
    all_labels1 = []
    all_preds2 = []
    all_labels2 = []
    all_preds3 = []
    all_probs3 = []
    all_labels3 = []
    all_preds4 = []
    all_labels4 = []
    eval_loss = 0.0

    with torch.no_grad():
        with tqdm(test_data, desc=f"Evaluating:") as t:
            for patient in t:
                _, y1, y2, y3 = patient[-1]
                y1 = torch.tensor(y1.values, dtype=torch.float32, device=device)
                y2_tensor_type = torch.long
                if args.dataset == 'tcm_lung':
                    y2_tensor_type = torch.float32
                y2 = torch.tensor(y2.values, dtype=y2_tensor_type, device=device)
                y3_binary = (y3 != 0).astype(float)
                y4 = torch.tensor(y3.values, dtype=torch.float32, device=device)
                y3 = torch.tensor(y3_binary.values, dtype=torch.float32, device=device)

                
                if 'Ours' in model_name:
                    output1, output2, output3, orth_loss, contrastive_loss = model(patient)
                    # y1是诊断，是多指标二分类
                    loss1 = criterions[0](output1, y1)
                    if args.dataset == 'Ours':
                        # y2是基础处方，是单指标多分类
                        loss2 = criterions[1](output2.view(1, -1), y2)
                    elif args.dataset == 'tcm_lung':
                        loss2 = criterions[1](output2, y2)
                    else:
                        raise ValueError('Invalid dataset.')
                else:
                    output3 = model(patient)
                # y3是个性化处方，是多指标二分类
                loss3 = criterions[2](output3, y3)
                # # y4是个性化处方剂量，是多标签回归 
                # loss4 = criterions[3](output4, y4)  

                if 'Ours' in model_name:
                    loss = (args.alpha1 * loss1) + (args.alpha2 * loss2) + loss3 + (args.alpha5 * orth_loss) + (args.alpha6 * contrastive_loss)
                else:
                    loss = loss3

                eval_loss += loss.item()

                
                # 使用 sigmoid 将 logits 转换为概率
                if 'Ours' in model_name:
                    # 使用 sigmoid 将 logits 转换为概率
                    probabilities1 = torch.sigmoid(output1)
                    probabilities3 = torch.sigmoid(output3)

                    # 设定阈值，将概率转为二分类预测值 (0 或 1)
                    preds1 = (probabilities1 > 0.5).float()
                    preds3 = (probabilities3 > 0.5).float()
                    if args.dataset == 'Ours':
                        probabilities2 = torch.softmax(output2, dim=0)
                        preds2 = torch.argmax(probabilities2, dim=0)
                    elif args.dataset == 'tcm_lung':
                        # TCM_Lung数据集treat可多选
                        probabilities2 = torch.sigmoid(output2)
                        preds2 = (probabilities2 > 0.5).float()
                    else:
                        raise ValueError('Invalid dataset.')
                else:
                    # 非Ours模型只处理output3，其他用0填充
                    probabilities3 = torch.sigmoid(output3)
                    preds3 = (probabilities3 > 0.5).float()
                    probabilities1 = torch.zeros_like(y1)
                    probabilities2 = torch.zeros_like(y2)
                    preds1 = torch.zeros_like(y1)
                    preds2 = torch.zeros_like(y2)

                # 将每个批次的结果添加到列表中
                all_preds1.append(preds1.cpu().numpy())
                all_labels1.append(y1.cpu().numpy())
                all_preds2.append(preds2.cpu().numpy())
                all_labels2.append(y2.cpu().numpy())
                all_probs3.append(probabilities3.cpu().numpy())
                all_preds3.append(preds3.cpu().numpy())
                all_labels3.append(y3.cpu().numpy())
                # all_preds4.append(preds4.cpu().numpy())
                # all_labels4.append(y4.cpu().numpy())

    # 数据格式转换
    all_labels1 = np.vstack(all_labels1)
    all_preds1 = np.vstack(all_preds1)
    if args.dataset == 'Ours':
        all_labels2 = np.array(all_labels2).flatten()
        all_preds2 = np.array(all_preds2).flatten()
    elif args.dataset == 'tcm_lung':
        all_labels2 = np.vstack(all_labels2)
        all_preds2 = np.vstack(all_preds2)
    else:
        raise ValueError('Invalid dataset.')
    all_labels3 = np.vstack(all_labels3)
    # all_preds3 = np.vstack(all_preds3)
    all_probs3 = np.vstack(all_probs3)
    # if 'dose' in model_name: 
    #     all_preds4 = np.vstack(all_preds4)
    #     all_labels4 = np.vstack(all_labels4)  

    # 计算y1/y2指标
    y1_acc, y1_f1, y1_spec, y1_sens = classification_evaluate_metrics(all_labels1, all_preds1, 'multilabel')
    y2_type = 'multiclass'
    if args.dataset == 'tcm_lung':
        y2_type = 'multilabel'
    y2_acc, y2_f1, y2_spec, y2_sens = classification_evaluate_metrics(all_labels2, all_preds2, y2_type)



    # 计算y3指标 
    y3_metrics = evaluate_y3_at_k(all_labels3, all_probs3)
    
    # # 计算y4指标 
    # y4_metrics = regression_evaluate_metrics(all_labels4, all_preds4)

    # 组织返回结果列表（共28个元素）
    return [
        # y1指标 (4)
        y1_acc, y1_f1, y1_spec, y1_sens,
        # y2指标 (4)
        y2_acc, y2_f1, y2_spec, y2_sens,
        # y3@5
        y3_metrics['f1@k'][0], y3_metrics['precision@k'][0], y3_metrics['recall@k'][0],
        # y3@10
        y3_metrics['f1@k'][1], y3_metrics['precision@k'][1], y3_metrics['recall@k'][1],
        # y3@15
        y3_metrics['f1@k'][2], y3_metrics['precision@k'][2], y3_metrics['recall@k'][2],
        # y3@20
        y3_metrics['f1@k'][3], y3_metrics['precision@k'][3], y3_metrics['recall@k'][3],
        # # y4指标 (3)
        # y4_metrics[0], y4_metrics[1], y4_metrics[2],
        # 平均损失
        eval_loss / len(test_data)
    ]

def training(model, model_name, num_epochs, train_data, test_data, criterions, optimizer, device, args=None):
    """训练模型并在验证集上评估（单标签多分类）"""
    best_accuracy = 0.0
    best_loss = float('inf')
    best_model_pt = None
    train_losses = []
    eval_losses = []
    patience_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0  # 记录每个epoch的总损失

        with tqdm(train_data, desc=f"Epoch [{epoch + 1}/{num_epochs}]") as t:
            for patient in t:
                optimizer.zero_grad()
                # 此处构建标签，只需要patient[-1]
                _, y1, y2, y3 = patient[-1]
                y1 = torch.tensor(y1.values, dtype=torch.float32, device=device)
                y2_tensor_type = torch.long
                if args.dataset == 'tcm_lung':
                    y2_tensor_type = torch.float32
                y2 = torch.tensor(y2.values, dtype=y2_tensor_type, device=device)
                y3_binary = (y3 != 0).astype(float)
                y4 = torch.tensor(y3.values, dtype=torch.float32, device=device)
                y3 = torch.tensor(y3_binary.values, dtype=torch.float32, device=device)

                # 模型前向传播
                if 'Ours' in model_name:
                    output1, output2, output3, orth_loss, contrastive_loss = model(patient)
                    
                    # Compute all losses
                    loss1 = criterions[0](output1, y1)
                    if args.dataset == 'Ours':
                        loss2 = criterions[1](output2.view(1, -1), y2)
                    elif args.dataset == 'tcm_lung':
                        # tcm_lung数据集y2可多选 所以损失函数选择BCEWithLogitsLoss
                        loss2 = criterions[1](output2, y2)
                    else:
                        raise ValueError('Invalid dataset.')
                    
                else:  # SimpleMLP
                    output3 = model(patient)
                loss3 = criterions[2](output3, y3)

                if 'Ours' in model_name:
                    loss = (args.alpha1 * loss1) + (args.alpha2 * loss2) + loss3 + (args.alpha5 * orth_loss) + (args.alpha6 * contrastive_loss)
                else:
                    loss = loss3

                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                epoch_loss += loss.item()
                t.set_postfix(loss=loss.item())  # 在进度条上显示当前batch的loss

        # 调用 evaluation 进行验证
        results = evaluation(model, model_name, test_data, criterions, device, args)
        eval_loss = results[-1]

        # 早停逻辑
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model_pt = model.state_dict()
            patience_counter = 0  # 重置计数器
            print(f"New best model! Val Loss: {best_loss:.4f}, y3_f1@15: {results[15]:.4f}")
        else:
            patience_counter += 1
            print(f"Val Loss did not improve ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        train_losses.append(epoch_loss / sum(len(patient) for patient in train_data))
        eval_losses.append(eval_loss)

    # 每个epoch都绘制一次，绘制损失曲线
    plot_losses(train_losses, eval_losses, model_name)
    return best_model_pt


def testing(model, model_name, test_data, criterions, device, args=None):
    """使用最佳模型进行测试并记录结果到CSV（增加多次抽样评估）"""
    print("\nTesting the best model with multiple sampling...")
    
    # 设置抽样次数和比例
    num_samples = 10
    sample_ratio = 0.8
    
    # 存储每次评估的结果
    all_results = []
    
    for i in range(num_samples):
        # 随机抽样
        sample_size = int(len(test_data) * sample_ratio)
        sampled_data = random.sample(test_data, sample_size)
        
        # 评估当前样本
        results = evaluation(model, model_name, sampled_data, criterions, device, args)
        all_results.append(results)
    
    # 计算均值和标准差
    all_results = np.array(all_results)
    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)
    
    # 打印结果
    print(f"\nEvaluation results over {num_samples} samples:")
    print(f"y3_f1@15: {mean_results[16]:.4f} ± {std_results[16]:.4f}")
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 解析结果
    headers = [
        "model_name", "timestamp",
        # y1指标
        "y1_acc", "y1_f1", "y1_spec", "y1_sens",
        # y2指标
        "y2_acc", "y2_f1", "y2_spec", "y2_sens",
        # y3@5
        "y3_f1@5", "y3_p@5", "y3_r@5",
        # y3@10
        "y3_f1@10", "y3_p@10", "y3_r@10",
        # y3@15
        "y3_f1@15", "y3_p@15", "y3_r@15",
        # y3@20
        "y3_f1@20", "y3_p@20", "y3_r@20",
        "avg_loss"
    ]
    
    # 构建数据行（格式化为mean ± std）
    row_data = [
        f"{model_name}_emb{args.embedding_dim}_a1{args.alpha1}_a2{args.alpha2}_a3{args.alpha3}_a5{args.alpha5}_a6{args.alpha6}",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # y1指标
        f"{mean_results[0]:.4f} ± {std_results[0]:.4f}",  # y1_acc
        f"{mean_results[1]:.4f} ± {std_results[1]:.4f}",  # y1_f1
        f"{mean_results[2]:.4f} ± {std_results[2]:.4f}",  # y1_spec
        f"{mean_results[3]:.4f} ± {std_results[3]:.4f}",  # y1_sens
        # y2指标
        f"{mean_results[4]:.4f} ± {std_results[4]:.4f}",  # y2_acc
        f"{mean_results[5]:.4f} ± {std_results[5]:.4f}",  # y2_f1
        f"{mean_results[6]:.4f} ± {std_results[6]:.4f}",  # y2_spec
        f"{mean_results[7]:.4f} ± {std_results[7]:.4f}",  # y2_sens
        # y3@5
        f"{mean_results[8]:.4f} ± {std_results[8]:.4f}",  # y3_f1@5
        f"{mean_results[9]:.4f} ± {std_results[9]:.4f}",  # y3_p@5
        f"{mean_results[10]:.4f} ± {std_results[10]:.4f}",  # y3_r@5
        # y3@10
        f"{mean_results[11]:.4f} ± {std_results[11]:.4f}",  # y3_f1@10
        f"{mean_results[12]:.4f} ± {std_results[12]:.4f}",  # y3_p@10
        f"{mean_results[13]:.4f} ± {std_results[13]:.4f}",  # y3_r@10
        # y3@15
        f"{mean_results[14]:.4f} ± {std_results[14]:.4f}",  # y3_f1@15
        f"{mean_results[15]:.4f} ± {std_results[15]:.4f}",  # y3_p@15
        f"{mean_results[16]:.4f} ± {std_results[16]:.4f}",  # y3_r@15
        # y3@20
        f"{mean_results[17]:.4f} ± {std_results[17]:.4f}",  # y3_f1@20
        f"{mean_results[18]:.4f} ± {std_results[18]:.4f}",  # y3_p@20
        f"{mean_results[19]:.4f} ± {std_results[19]:.4f}",  # y3_r@20
        # avg_loss
        f"{mean_results[-1]:.4f} ± {std_results[-1]:.4f}"
    ]
    
    # 写入CSV文件
    with open("results/results.csv", "a+", newline='') as f:
        f.seek(0)
        if len(f.read(1)) == 0:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        writer = csv.writer(f)
        writer.writerow(row_data)
    
    print(f"\nResults saved to results/results.csv")
    return mean_results[16]  # 返回 y3_f1@15 的均值