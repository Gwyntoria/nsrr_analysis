import logging
import os
from pathlib import Path
from dataclasses import dataclass

# Logging configuration
LOG_LEVEL = logging.INFO

VERSION_MAJOR = 0
VERSION_MINOR = 2
VERSION_PATCH = 0
VERSION = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"

MODEL_NAME = f"ssc_model_{VERSION}.pth"
DATASET_NAME = "shhs"  # mesa, shhs


@dataclass
class PathConfig:
    # Input directories
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    dataset_dir: str = os.path.join(base_dir, "data", DATASET_NAME)
    model_save_dir: str = os.path.join(base_dir, "models")
    # Output directories
    plots_dir: str = os.path.join(base_dir, "plots", VERSION)
    logs_dir: str = os.path.join(base_dir, "logs", VERSION)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    patience: int = 10
    lr_patience: int = 5
    sequence_length: int = 32
    hidden_size: int = 128
    num_layers: int = 3
    num_classes: int = 4

    input_size: int = 5
    dropout: float = 0.5


def setup_directories():
    """创建必要的目录结构"""
    directories = [
        PathConfig.dataset_dir,
        PathConfig.model_save_dir,
        PathConfig.plots_dir,
        PathConfig.logs_dir,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def setup_logger(name=None, log_file=None, level=logging.DEBUG):
    """配置并返回一个 logger 对象

    Args:
        name (string, optional): 日志记录器名称. Defaults to None.
        log_file (string, optional): 日志文件路径. Defaults to None.
        level (int, optional): 日志级别. Defaults to logging.DEBUG.

    Returns:
        Logger: 配置好的 logger 对象
    """
    # 创建日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 获取 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加 Handler
    if not logger.handlers:
        # 控制台日志处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件日志处理器（可选）
        if log_file:
            # 如果提供了日志文件路径，确保其父目录存在
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
