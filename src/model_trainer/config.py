import logging
import os
from pathlib import Path

# Logging configuration
LOG_LEVEL = logging.INFO

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data", "mesa")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")

# Model configuration
MODEL_VERSION = "v0.1"
MODEL_NAME = f"ssc_model_{MODEL_VERSION}.pth"

# Output directories
PLOTS_DIR = os.path.join(BASE_DIR, "plots", MODEL_VERSION)
LOGS_DIR = os.path.join(BASE_DIR, "logs", MODEL_VERSION)

def setup_directories():
    """创建必要的目录结构"""
    directories = [DATA_DIR, MODEL_SAVE_DIR, PLOTS_DIR, LOGS_DIR]
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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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
