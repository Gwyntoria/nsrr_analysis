import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb  # 用于实验跟踪
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import evaluate_model, plot_training_history

from data_loader import SleepDataset
from model import SleepStageClassifier
from config import (
    LOG_LEVEL,
    MODEL_NAME,
    PATH_CONFIG,
    TRAINING_CONFIG,
    TrainingConfig,
    setup_directories,
    setup_logger,
)

# Configure logging
logger = setup_logger(
    name=__name__,
    log_file=os.path.join(PATH_CONFIG.logs_dir),
    level=LOG_LEVEL,
)


def train_model(
    config: TrainingConfig, data_dir: str = None, model_save_dir: str = None
):
    if data_dir is None or model_save_dir is None:
        logger.error("Data directory and model save directory must be provided")
        raise ValueError("Data directory and model save directory must be provided")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    try:
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

        logger.info(f"Using data directory: {data_dir}")
        logger.info(f"Using model save directory: {model_save_dir}")

        # 初始化wandb
        wandb.init(project="sleep-stage-classification")

        # 准备数据
        dataset = SleepDataset(
            data_dir, sequence_length=config.sequence_length, augment=True
        )

        # 计算划分大小
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        # 直接划分数据集
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # 初始化模型
        model = SleepStageClassifier(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
        ).to(device)

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
            lr=config.learning_rate,
            weight_decay=config.weight_decay,  # 增加正则化
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # 使用余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # 初始周期
            T_mult=2,  # 每次重启后周期翻倍
            eta_min=config.learning_rate / 100,  # 最小学习率
        )

        # 增加早停的容忍度
        early_stopping_counter = 0
        best_val_loss = float("inf")

        # 添加学习率衰减的早停
        lr_counter = 0
        best_lr_loss = float("inf")

        # 记录训练历史
        train_losses = []
        val_losses = []

        # 训练循环
        for epoch in range(config.epochs):
            model.train()
            total_train_loss = 0

            # 添加进度条
            train_pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]"
            )
            for features, labels in train_pbar:
                features = features.float().to(device)
                labels = labels.long().to(device)

                optimizer.zero_grad()
                # 修改模型输出的解包
                main_output, _, attention_weights = model(features)

                # 只计算分类损失
                loss = criterion(main_output, labels)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_pbar.set_postfix({"loss": loss.item()})

                # 记录到wandb
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )

            # 验证
            model.eval()
            total_val_loss = 0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Val]")
            with torch.no_grad():
                for features, labels in val_pbar:
                    features = features.float().to(device)
                    labels = labels.long().to(device)

                    # 获取模型输出
                    main_output, _, attention_weights = model(features)

                    # 只计算分类损失
                    loss = criterion(main_output, labels)

                    total_val_loss += loss.item()
                    val_pbar.set_postfix({"val_loss": loss.item()})

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # 验证后的早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(model_save_dir, MODEL_NAME))
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # 在训练循环中添加学习率调整
            if avg_val_loss < best_lr_loss:
                best_lr_loss = avg_val_loss
                lr_counter = 0
            else:
                lr_counter += 1

            if lr_counter >= config.lr_patience:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5
                lr_counter = 0
                logger.info(
                    f"Reducing learning rate to {optimizer.param_groups[0]['lr']}"
                )

            # 记录更多指标
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }
            )

            logger.info(f"Epoch [{epoch + 1}/{config.epochs}]")
            logger.info(f"Training Loss: {avg_train_loss:.4f}")
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        # 绘制训练历史
        plot_training_history(train_losses, val_losses, plot_save_dir=PATH_CONFIG.plots_dir)

        # 评估最终模型
        accuracy = evaluate_model(model, val_loader, device, plot_save_dir=PATH_CONFIG.plots_dir)
        print(f"Final Validation Accuracy: {accuracy:.4f}")

        return accuracy

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def save_checkpoint(model, optimizer, epoch, loss, model_save_dir, model_name):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    checkpoint_path = os.path.join(model_save_dir, f"checkpoint_{model_name}")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, model_save_dir, model_name):
    checkpoint_path = os.path.join(model_save_dir, f"checkpoint_{model_name}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        logger.info(f"Resumed from epoch {start_epoch}")
        return start_epoch, loss
    return 0, float("inf")


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.float().to(device)
            labels = labels.long().to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = sum(p == label for p, label in zip(all_preds, all_labels)) / len(
        all_preds
    )

    return avg_val_loss, accuracy


if __name__ == "__main__":
    setup_directories()
    try:
        train_model(
            TRAINING_CONFIG, PATH_CONFIG.dataset_dir, PATH_CONFIG.model_save_dir
        )
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
