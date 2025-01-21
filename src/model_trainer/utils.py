import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from log_config import setup_logger
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from tqdm import tqdm

logger = setup_logger(name="trainer", log_file="training.log")


def plot_training_history(train_losses, val_losses, save_dir="../../plots"):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图片
    plt.savefig(os.path.join(save_dir, "training_history.png"))
    plt.close()


def evaluate_model(model, test_loader, device, save_dir="../../plots"):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_probs = []  # 存储预测概率
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features = features.float().to(device)
            labels = labels.to(device)
            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算每个类别的ROC AUC
    roc_auc = {}
    for i in range(6):  # 6个睡眠阶段
        roc_auc[f"Stage_{i}"] = roc_auc_score((all_labels == i).astype(int), all_probs[:, i], average="macro")

    # 计算平均精确率
    avg_precision = {}
    for i in range(6):
        avg_precision[f"Stage_{i}"] = average_precision_score((all_labels == i).astype(int), all_probs[:, i])

    # 绘制每个类别的PR曲线
    plt.figure(figsize=(12, 8))
    for i in range(6):
        precision, recall, _ = precision_recall_curve((all_labels == i).astype(int), all_probs[:, i])
        plt.plot(recall, precision, label=f"Stage {i} (AP={avg_precision[f'Stage_{i}']:.2f})")

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

    # 修改类别名称
    stage_names = ["Wake", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "REM"]

    # 打印详细的评估报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=stage_names))

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=stage_names, yticklabels=stage_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # 保存混淆矩阵图片
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    return correct / total
