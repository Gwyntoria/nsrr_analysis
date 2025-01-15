import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_loader import SleepDataset
from model import SleepStageClassifier
from utils import evaluate_model, plot_training_history


def train_model(data_dir, model_save_dir, epochs=50, batch_size=32, learning_rate=0.001):
    # 准备数据
    dataset = SleepDataset(os.path.join(data_dir, "sleep_data.csv"))

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SleepStageClassifier().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    # 记录训练历史
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for features, labels in train_loader:
            features = features.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.float().to(device)
                labels = labels.long().to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 学习率调整
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

    # 绘制训练历史
    plot_training_history(train_losses, val_losses)

    # 评估最终模型
    accuracy = evaluate_model(model, val_loader, device)
    print(f"Final Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    data_dir = "/data/mesa/csvs"
    model_save_dir = "/models"
    train_model(data_dir, model_save_dir)
