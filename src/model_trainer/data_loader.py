import logging
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# 配置日志
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class SleepDataset(Dataset):
    def __init__(self, data_dir, sequence_length=32):
        """
        初始化数据集
        Args:
            data_dir: 包含所有睡眠数据CSV文件的目录路径
            sequence_length: 序列长度，即每个样本包含多少个时间步
        """
        logger.info(f"Initializing SleepDataset with directory: {data_dir}")
        logger.info(f"Sequence length: {sequence_length}")

        self.sequence_length = sequence_length
        self.data = self.load_all_data(data_dir)
        self.process_data()
        self.prepare_sequences()

    def load_all_data(self, data_dir):
        """加载目录下的所有CSV文件并合并数据"""
        all_data = []
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        logger.info(f"Found {len(csv_files)} CSV files in directory")

        for file_name in csv_files:
            file_path = os.path.join(data_dir, file_name)
            logger.debug(f"Loading file: {file_path}")

            df = pd.read_csv(file_path)
            user_id = file_name.split(".")[0].split("-")[-1]
            logger.debug(f"Processing user_id: {user_id}, data shape: {df.shape}")

            # 重命名列名以匹配代码
            df = df.rename(columns={"timestamp": "time", "heart_rate": "heart_rate", "sleep_stage": "sleep_stage"})
            df["user_id"] = user_id
            all_data.append(df)

        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total combined data shape: {combined_data.shape}")
        return combined_data

    def process_data(self):
        """数据预处理"""
        logger.info("Starting data preprocessing")
        initial_size = len(self.data)

        # 将时间戳转换为datetime对象
        logger.debug("Converting timestamps to datetime")
        self.data["time"] = pd.to_datetime(self.data["time"], unit="s")

        # 提取时间特征
        logger.debug("Extracting time features")
        self.data["hour"] = self.data["time"].dt.hour
        self.data["minute"] = self.data["time"].dt.minute

        # 将时间转换为循环特征
        logger.debug("Creating cyclical time features")
        self.data["hour_sin"] = np.sin(2 * np.pi * self.data["hour"] / 24)
        self.data["hour_cos"] = np.cos(2 * np.pi * self.data["hour"] / 24)
        self.data["minute_sin"] = np.sin(2 * np.pi * self.data["minute"] / 60)
        self.data["minute_cos"] = np.cos(2 * np.pi * self.data["minute"] / 60)

        # 计算心率变化特征
        logger.debug("Computing heart rate derivatives")
        self.data["heart_rate_diff"] = self.data.groupby("user_id")["heart_rate"].diff()
        self.data["heart_rate_diff2"] = self.data.groupby("user_id")["heart_rate_diff"].diff()

        # 对每个用户的特征分别进行归一化
        logger.debug("Normalizing features per user")
        features_to_normalize = ["heart_rate", "heart_rate_diff", "heart_rate_diff2"]
        for feature in features_to_normalize:
            self.data[feature] = self.data.groupby("user_id")[feature].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

        # 确保所有数据都是有效的
        logger.debug("Removing invalid data")
        self.data = self.data.dropna()

        # 记录数据清理的影响
        removed_rows = initial_size - len(self.data)
        logger.info(f"Removed {removed_rows} invalid rows ({removed_rows / initial_size * 100:.2f}% of data)")

        # 将睡眠分期映射到数值
        logger.debug("Mapping sleep stages")
        stage_mapping = {
            0: 0,  # wake-stage
            1: 1,  # sleep-stage1
            2: 2,  # sleep-stage2
            3: 3,  # sleep-stage3
            4: 4,  # sleep-stage4
            5: 5,  # REM
        }
        self.data["sleep_stage"] = self.data["sleep_stage"].map(stage_mapping)

        # 检查睡眠阶段分布
        stage_dist = self.data["sleep_stage"].value_counts()
        logger.info("Sleep stage distribution:")
        for stage, count in stage_dist.items():
            logger.info(f"Stage {stage}: {count} samples ({count / len(self.data) * 100:.2f}%)")

    def prepare_sequences(self):
        """准备序列数据"""
        logger.info("Preparing sequences")
        sequences = []
        labels = []

        feature_columns = [
            "heart_rate",
            "heart_rate_diff",
            "heart_rate_diff2",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
        ]

        # 按用户和时间排序
        self.data = self.data.sort_values(["user_id", "time"])

        total_sequences = 0
        invalid_sequences = 0

        for user_id in self.data["user_id"].unique():
            user_data = self.data[self.data["user_id"] == user_id]
            logger.debug(f"Processing sequences for user {user_id}, data points: {len(user_data)}")

            features = user_data[feature_columns].values
            targets = user_data["sleep_stage"].values

            # 创建滑动窗口序列
            for i in range(len(user_data) - self.sequence_length + 1):
                total_sequences += 1
                seq = features[i : i + self.sequence_length]
                label = targets[i + self.sequence_length - 1]

                # 确保序列中没有无效值
                if not np.isnan(seq).any() and not np.isnan(label):
                    sequences.append(seq)
                    labels.append(label)
                else:
                    invalid_sequences += 1

        self.sequences = np.array(sequences)
        self.labels = np.array(labels)

        logger.info(f"Created {len(sequences)} valid sequences")
        logger.info(f"Dropped {invalid_sequences} invalid sequences ({invalid_sequences / total_sequences * 100:.2f}%)")
        logger.info(f"Final sequence shape: {self.sequences.shape}")
        logger.info(f"Final labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label
