import logging
import os

import torch
import torch.nn as nn
import wandb  # 用于实验跟踪
from data_loader import SleepDataset
from model import SleepStageClassifier
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import evaluate_model, plot_training_history

model_version = "v0.1"
model_name = f"ssc_model_{model_version}.pth"
data_dir = "../../data/mesa"
model_save_dir = "../../models"


def train_model(data_dir, model_save_dir, epochs=150, batch_size=64, learning_rate=0.0005):
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # 初始化wandb
    wandb.init(project="sleep-stage-classification")

    # 准备数据
    train_dataset = SleepDataset(data_dir, sequence_length=32, training=True)  # 训练集
    val_dataset = SleepDataset(data_dir, sequence_length=32, training=False)   # 验证集

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SleepStageClassifier(hidden_size=128, num_layers=3).to(device)

    # 计算类别权重
    labels = [label for _, label in train_dataset]
    # 将标签转换为长整型
    labels = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    # 使用加权交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 使用AdamW优化器，调整参数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # 增加正则化
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 使用余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 初始周期
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=learning_rate/100  # 最小学习率
    )

    # 增加早停的容忍度
    patience = 20  # 从15增加到20
    early_stopping_counter = 0
    best_val_loss = float("inf")
    
    # 添加学习率衰减的早停
    lr_patience = 5
    lr_counter = 0
    best_lr_loss = float("inf")

    # 记录训练历史
    train_losses = []
    val_losses = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # 添加进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for features, labels in train_pbar:
            features = features.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            # 记录梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})

            # 记录到wandb
            wandb.log({"batch_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0], "gradient_norm": grad_norm})

        # 验证
        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
        with torch.no_grad():
            for features, labels in val_pbar:
                features = features.float().to(device)
                labels = labels.long().to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 验证后的早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            # 保存最佳模型
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(model.state_dict(), os.path.join(model_save_dir, model_name))
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # 在训练循环中添加学习率调整
        if avg_val_loss < best_lr_loss:
            best_lr_loss = avg_val_loss
            lr_counter = 0
        else:
            lr_counter += 1
            
        if lr_counter >= lr_patience:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            lr_counter = 0
            logger.info(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")

        # 记录更多指标
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )

        logger.info(f"Epoch [{epoch + 1}/{epochs}]")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

    # 绘制训练历史
    plot_training_history(train_losses, val_losses)

    # 评估最终模型
    accuracy = evaluate_model(model, val_loader, device)
    print(f"Final Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_model(
        data_dir=data_dir,
        model_save_dir=model_save_dir,
        epochs=200,  # 增加最大轮数
        batch_size=64,  # 减小batch_size以增加随机性
        learning_rate=0.0005,  # 降低初始学习率
    )
