import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from config import PATH_CONFIG, LOG_LEVEL, setup_logger

# 配置日志
logger = setup_logger(
    name=__name__,
    log_file=os.path.join(PATH_CONFIG.logs_dir),
    level=LOG_LEVEL,
)


class SleepDataset(Dataset):
    def __init__(self, data_dir, sequence_length=8, augment=True):
        """
        初始化数据集
        Args:
            data_dir: 包含所有睡眠数据CSV文件的目录路径
            sequence_length: 序列长度，即每个样本包含多少个时间步
            augment: 是否进行数据增强
        """
        logger.info(f"Sequence length: {sequence_length}")

        self.sequences = None
        self.labels = None

        self.sequence_length = sequence_length
        self.augment = augment
        self.data = self.load_all_data(data_dir)
        self.process_data()
        self.prepare_sequences()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # 对序列进行数据增强
        sequence = self._augment_sequence(sequence)
        return sequence, label

    def load_all_data(self, data_dir: str) -> pd.DataFrame:
        """加载目录下的所有CSV文件并合并数据"""
        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        logger.info(f"Loading data from directory: {data_dir}")

        all_data = []
        csv_files = []

        for root, _, files in os.walk(data_dir):
            csv_files.extend(
                [os.path.join(root, f) for f in files if f.endswith(".csv")]
            )
        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            logger.debug(f"Loading file: {file_path}")

            df = pd.read_csv(file_path)
            # 将所有列名转换为小写并去除空格
            df.columns = df.columns.str.lower().str.strip()
            logger.debug(f"Columns in {file}: {df.columns.tolist()}")

            # 重命名列名以匹配代码
            df = df.rename(
                columns={
                    "timestamp": "time",
                    "hr": "heart_rate",
                    "sleep_stage": "sleep_stage",
                }
            )

            # 确保只保留需要的列
            required_columns = ["time", "heart_rate", "sleep_stage"]
            df = df[required_columns]

            user_id = file.split(".")[0].split("/")[-1]
            logger.debug(f"Processing user_id: {user_id}, data shape: {df.shape}")
            df["user_id"] = user_id
            all_data.append(df)

        # 合并所有加载的CSV数据到一个DataFrame中
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data columns: {combined_data.columns.tolist()}")
        logger.info(f"Total combined data shape: {combined_data.shape}")
        return combined_data

    def process_data(self):
        """数据预处理"""
        logger.info("Starting data preprocessing")
        # initial_size = len(self.data)

        # 将时间戳转换为相对时间特征
        logger.info("Processing timestamps")
        self.data = self.data.sort_values(["user_id", "time"])

        # 对每个用户分别处理
        for user_id in self.data["user_id"].unique():
            logger.debug(f"Processing user_id: {user_id}")
            user_mask = self.data["user_id"] == user_id

            # 计算相对时间（相对于每个用户数据的开始时间）
            start_time = self.data[user_mask]["time"].min()
            logger.debug(f"Start time for user_id {user_id}: {start_time}")
            self.data.loc[user_mask, "relative_time"] = (
                self.data[user_mask]["time"] - start_time
            )

            # 将数据缩减为5分钟采样一次
            self.data.loc[user_mask, "time_5min"] = (
                self.data[user_mask]["time"] // 300 * 300
            )

            # 获取用户数据副本进行分组处理
            user_data = self.data[user_mask].copy()

            grouped_data = (
                user_data.groupby(["user_id", "time_5min"])
                .agg(
                    heart_rate=("heart_rate", "mean"),
                    sleep_stage=("sleep_stage", "max"),
                    relative_time=("relative_time", "first"),
                )
                .reset_index()
            )
            logger.debug(f"Grouped data for user_id {user_id}: {grouped_data.shape}")

            # 正确处理数据替换 - 使用merge而不是直接赋值
            # 移除用户数据
            self.data = self.data[~user_mask]
            # 将分组后的数据添加回主数据框
            self.data = pd.concat([self.data, grouped_data], ignore_index=True)

            # 更新用户掩码
            user_mask = self.data["user_id"] == user_id

            # 计算心率的统计特征 - 针对每个用户单独计算
            self.data.loc[user_mask, "heart_rate_diff"] = self.data.loc[
                user_mask, "heart_rate"
            ].diff()
            # 确保填充NaN值
            self.data.loc[user_mask, "heart_rate_diff"] = self.data.loc[
                user_mask, "heart_rate_diff"
            ].fillna(0)

            self.data.loc[user_mask, "heart_rate_ma"] = (
                self.data.loc[user_mask, "heart_rate"]
                .rolling(window=5, min_periods=1)
                .mean()
            )
            # 确保填充NaN值
            self.data.loc[user_mask, "heart_rate_ma"] = self.data.loc[
                user_mask, "heart_rate_ma"
            ].fillna(self.data.loc[user_mask, "heart_rate"])

            # 标准化、归一化特征
            for feature in [
                "heart_rate",
                "heart_rate_diff",
                "heart_rate_ma",
                "relative_time",
            ]:
                # 检查是否有有效数据进行归一化
                if self.data.loc[user_mask, feature].notna().any():
                    mean = self.data.loc[user_mask, feature].mean()
                    std = self.data.loc[user_mask, feature].std()
                    # 防止除零错误
                    if std == 0 or pd.isna(std):
                        std = 1e-8
                    self.data.loc[user_mask, feature] = (
                        self.data.loc[user_mask, feature] - mean
                    ) / (std + 1e-8)
                    logger.debug(
                        f"Feature {feature} normalized for user_id {user_id}: mean={mean}, std={std}"
                    )
                else:
                    logger.warning(
                        f"No valid data for feature {feature} for user_id {user_id}"
                    )

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

        # 选择特征列
        feature_columns = [
            "relative_time",
            "heart_rate",
            "heart_rate_diff",
            "heart_rate_ma",
        ]

        # 按用户和时间排序
        self.data = self.data.sort_values(["user_id", "time"])

        # 对每个用户的数据进行处理
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

    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """对数据序列进行增强
        1. 随机添加噪声
        2. 随机时间扭曲
        3. 随机掩码

        Args:
            sequence (np.ndarray): 需要增强的数据序列

        Returns:
            np.ndarray: 增强后的数据序列
        """
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
