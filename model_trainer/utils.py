import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm
import pandas as pd


def plot_training_history(train_losses, val_losses, save_dir):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 确保保存目录存在
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "training_history")
    os.makedirs(save_dir, exist_ok=True)

    # 保存图片
    plt.savefig(os.path.join(save_dir, "training_history.png"))
    plt.close()


def evaluate_model(model, test_loader, device, save_dir):
    """评估模型性能"""
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features = features.float().to(device)
            labels = labels.to(device)

            # 获取主要输出
            main_output, _, _ = model(features)
            probs = F.softmax(main_output, dim=1)
            _, predicted = torch.max(main_output.data, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 修改类别名称
    stage_names = ["Wake", "Light", "Deep", "REM"]  # 更新为4分类的名称

    # 计算每个类别的ROC AUC
    roc_auc = {}
    for i in range(4):  # 修改为4个类别
        roc_auc[f"Stage_{i}"] = roc_auc_score(
            (all_labels == i).astype(int), all_probs[:, i], average="macro"
        )

    # 计算平均精确率
    avg_precision = {}
    for i in range(4):  # 修改为4个类别
        avg_precision[f"Stage_{i}"] = average_precision_score(
            (all_labels == i).astype(int), all_probs[:, i]
        )

    # 绘制每个类别的PR曲线
    plt.figure(figsize=(12, 8))
    stage_names_dict = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
    for i in range(4):  # 修改为4个类别
        precision, recall, _ = precision_recall_curve(
            (all_labels == i).astype(int), all_probs[:, i]
        )
        plt.plot(
            recall,
            precision,
            label=f"{stage_names_dict[i]} (AP={avg_precision[f'Stage_{i}']:.2f})",
        )

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "pr_curves.png"))
    plt.close()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.float().to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 打印详细的评估报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=stage_names))

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=stage_names,
        yticklabels=stage_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # 保存混淆矩阵图片
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    return correct / total


def predict_sleep_stages(model, time_series, heart_rates, device, sequence_length=32):
    """
    预测睡眠阶段
    Args:
        model: 训练好的模型
        time_series: 时间戳序列
        heart_rates: 心率序列
        device: 计算设备
        sequence_length: 序列长度
    Returns:
        预测的睡眠阶段序列
    """
    model.eval()
    predictions = []

    # 数据预处理
    time_series = np.array(time_series)
    heart_rates = np.array(heart_rates)

    # 计算特征
    relative_time = time_series - time_series[0]
    heart_rate_diff = np.diff(heart_rates, prepend=heart_rates[0])
    heart_rate_ma = (
        pd.Series(heart_rates).rolling(window=5, min_periods=1).mean().values
    )

    # 标准化特征
    features = np.column_stack(
        [relative_time, heart_rates, heart_rate_diff, heart_rate_ma]
    )

    # 使用滑动窗口进行预测
    with torch.no_grad():
        for i in range(len(features) - sequence_length + 1):
            sequence = features[i : i + sequence_length]

            # 添加前一个预测作为特征
            prev_stage = -1 if i == 0 else predictions[-1]
            sequence = np.column_stack([sequence, np.full(sequence_length, prev_stage)])

            # 转换为张量
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

            # 预测
            main_output, _, _ = model(sequence_tensor)
            pred = torch.argmax(main_output, dim=1).item()
            predictions.append(pred)

    # 处理序列开始的部分
    initial_predictions = [predictions[0]] * (sequence_length - 1)
    return initial_predictions + predictions
