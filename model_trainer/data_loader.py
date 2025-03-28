import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from config import PATH_CONFIG, LOG_LEVEL, setup_logger

# 配置日志
logger = setup_logger(
    name=__name__,
    log_file=os.path.join(PATH_CONFIG.logs_dir, "training.log"),
    level=LOG_LEVEL,
)


class SleepDataset(Dataset):
    def __init__(self, data_dir, sequence_length=32, augment=True):
        """
        初始化数据集
        Args:
            data_dir: 包含所有睡眠数据CSV文件的目录路径
            sequence_length: 序列长度，即每个样本包含多少个时间步
            augment: 是否进行数据增强
        """
        logger.info(f"Initializing SleepDataset with directory: {data_dir}")
        logger.info(f"Sequence length: {sequence_length}")

        self.sequence_length = sequence_length
        self.augment = augment
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
            df = df.rename(
                columns={
                    "timestamp": "time",
                    "HR": "heart_rate",
                    "sleep_stage": "sleep_stage",
                }
            )
            df["user_id"] = user_id
            all_data.append(df)

        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total combined data shape: {combined_data.shape}")
        return combined_data

    def process_data(self):
        """数据预处理"""
        logger.info("Starting data preprocessing")
        initial_size = len(self.data)

        # 将时间戳转换为相对时间特征
        logger.debug("Processing timestamps")
        self.data = self.data.sort_values(["user_id", "time"])

        # 对每个用户分别处理
        for user_id in self.data["user_id"].unique():
            user_mask = self.data["user_id"] == user_id
            # 计算相对时间（相对于每个用户数据的开始时间）
            start_time = self.data[user_mask]["time"].min()
            self.data.loc[user_mask, "relative_time"] = (
                self.data[user_mask]["time"] - start_time
            )

            # 计算心率的统计特征
            self.data.loc[user_mask, "heart_rate_diff"] = self.data[user_mask][
                "heart_rate"
            ].diff()
            self.data.loc[user_mask, "heart_rate_ma"] = (
                self.data[user_mask]["heart_rate"]
                .rolling(window=5, min_periods=1)
                .mean()
            )

            # 归一化特征
            for feature in [
                "heart_rate",
                "heart_rate_diff",
                "heart_rate_ma",
                "relative_time",
            ]:
                mean = self.data.loc[user_mask, feature].mean()
                std = self.data.loc[user_mask, feature].std()
                self.data.loc[user_mask, feature] = (
                    self.data.loc[user_mask, feature] - mean
                ) / (std + 1e-8)

        # 修改睡眠分期映射
        stage_mapping = {
            0: 0,  # wake
            1: 1,  # light
            2: 1,  # light
            3: 2,  # deep
            4: 2,  # deep
            5: 3,  # REM
        }
        self.data["sleep_stage"] = self.data["sleep_stage"].map(stage_mapping)

        # 检查睡眠阶段分布
        stage_dist = self.data["sleep_stage"].value_counts()
        stage_names = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
        logger.info("Sleep stage distribution:")
        for stage, count in stage_dist.items():
            logger.info(
                f"{stage_names[stage]}: {count} samples ({count/len(self.data)*100:.2f}%)"
            )

    def prepare_sequences(self):
        """准备序列数据"""
        logger.info("Preparing sequences")
        sequences = []
        labels = []

        # 更新特征列表，删除 prev_sleep_stage
        feature_columns = [
            "relative_time",
            "heart_rate",
            "heart_rate_diff",
            "heart_rate_ma",
        ]

        # 按用户和时间排序
        self.data = self.data.sort_values(["user_id", "time"])

        for user_id in self.data["user_id"].unique():
            user_data = self.data[self.data["user_id"] == user_id]

            features = user_data[feature_columns].values
            targets = user_data["sleep_stage"].values

            # 创建滑动窗口序列
            for i in range(len(user_data) - self.sequence_length + 1):
                seq = features[i : i + self.sequence_length]
                label = targets[i + self.sequence_length - 1]

                if not np.isnan(seq).any() and not np.isnan(label):
                    sequences.append(seq)
                    labels.append(label)

        self.sequences = np.array(sequences)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label

    def _augment_sequence(self, sequence):
        """增强的数据增强方法"""
        if not self.augment:
            return sequence

        try:
            # 随机添加噪声
            if np.random.random() < 0.5:
                noise_level = np.random.uniform(0.005, 0.015)
                noise = np.random.normal(0, noise_level, sequence.shape)
                sequence = sequence + noise

            # 随机时间扭曲
            if np.random.random() < 0.3:
                scale = np.random.uniform(0.85, 1.15)
                time_steps = np.arange(len(sequence))
                new_time_steps = (
                    np.linspace(0, len(sequence) - 1, len(sequence)) * scale
                )

                warped_sequence = np.zeros_like(sequence)
                for i in range(sequence.shape[1]):
                    warped_sequence[:, i] = np.interp(
                        time_steps, new_time_steps, sequence[:, i]
                    )
                sequence = warped_sequence

            # 添加随机掩码
            if np.random.random() < 0.2:
                mask_length = np.random.randint(1, 5)
                start_idx = np.random.randint(0, len(sequence) - mask_length)
                sequence[start_idx : start_idx + mask_length] = 0

            return sequence

        except Exception as e:
            logger.error(f"Error in data augmentation: {str(e)}")
            logger.error(f"Sequence shape: {sequence.shape}")
            raise e
