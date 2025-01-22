import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    MODEL_NAME,
    MODEL_SAVE_DIR,
    BASE_DIR,
    MODEL_VERSION,
    MODEL_CONFIG,
)
from model import SleepStageClassifier
from utils import predict_sleep_stages


def test_single_file(data_dir, model_path, result_dir):
    """
    使用单个文件测试模型性能
    Args:
        data_dir: 数据目录路径
        model_path: 模型文件路径
        result_dir: 结果保存目录
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)

    # 随机选择一个CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    test_file = random.choice(csv_files)
    print(f"Selected test file: {test_file}")

    # 读取数据
    df = pd.read_csv(os.path.join(data_dir, test_file))
    print(f"Data shape: {df.shape}")

    # 映射睡眠阶段
    stage_mapping = {
        0: 0,  # wake
        1: 1,  # light
        2: 1,  # light
        3: 2,  # deep
        4: 2,  # deep
        5: 3,  # REM
    }
    df["sleep_stage"] = df["sleep_stage"].map(stage_mapping)
    print("Sleep stage distribution after mapping:")
    print(df["sleep_stage"].value_counts())

    # 使用配置创建模型
    input_size = MODEL_CONFIG["input_size"]
    hidden_size = MODEL_CONFIG["hidden_size"]
    num_layers = MODEL_CONFIG["num_layers"]
    num_classes = MODEL_CONFIG["num_classes"]
    dropout = MODEL_CONFIG["dropout"]
    model = SleepStageClassifier(
        input_size, hidden_size, num_layers, num_classes, dropout
    ).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 预测睡眠阶段
    predictions = predict_sleep_stages(
        model, df["timestamp"].values, df["heart_rate"].values, device
    )

    # 将预测结果添加到DataFrame
    df["predicted_stage"] = predictions

    # 绘制对比图
    plt.figure(figsize=(15, 10))
    plt.plot(df["timestamp"], df["sleep_stage"], label="Actual", alpha=0.7)
    plt.plot(
        df["timestamp"], df["predicted_stage"], label="Predicted", alpha=0.7
    )
    plt.title("Sleep Stage Comparison")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Sleep Stage")
    plt.yticks([0, 1, 2, 3], ["Wake", "Light", "Deep", "REM"])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "stage_comparison.png"))
    plt.close()

    # 计算混淆矩阵
    cm = confusion_matrix(df["sleep_stage"], df["predicted_stage"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Wake", "Light", "Deep", "REM"],
        yticklabels=["Wake", "Light", "Deep", "REM"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    # 生成分类报告
    report = classification_report(
        df["sleep_stage"],
        df["predicted_stage"],
        target_names=["Wake", "Light", "Deep", "REM"],
        labels=[0, 1, 2, 3],
    )

    # 保存分类报告
    with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("\nClassification Report:")
    print(report)

    # 计算准确率
    accuracy = (df["sleep_stage"] == df["predicted_stage"]).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # 绘制每个阶段的准确率随时间的变化
    window_size = 100  # 滑动窗口大小
    rolling_accuracy = []
    timestamps = []

    for i in range(0, len(df) - window_size, window_size // 2):
        window_accuracy = (
            df["sleep_stage"].iloc[i : i + window_size]
            == df["predicted_stage"].iloc[i : i + window_size]
        ).mean()
        rolling_accuracy.append(window_accuracy)
        timestamps.append(df["timestamp"].iloc[i])

    plt.figure(figsize=(15, 5))
    plt.plot(timestamps, rolling_accuracy)
    plt.title("Rolling Accuracy Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "rolling_accuracy.png"))
    plt.close()

    # 保存测试结果到CSV
    df.to_csv(os.path.join(result_dir, "test_results.csv"), index=False)

    return accuracy


if __name__ == "__main__":
    # 设置路径
    data_dir = os.path.join(BASE_DIR, "data", "shhs", "shhs1")
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    result_dir = os.path.join(BASE_DIR, "results", MODEL_VERSION)

    # 运行测试
    accuracy = test_single_file(data_dir, model_path, result_dir)
