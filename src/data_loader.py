
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, data_path):
        """
        初始化数据集
        Args:
            data_path: 包含心率和睡眠分期的CSV文件路径
        """
        self.data = pd.read_csv(data_path)

        # 数据预处理
        self.process_data()

    def process_data(self):
        """数据预处理"""
        # 将时间转换为特征
        self.data["time"] = pd.to_datetime(self.data["time"])
        self.data["hour"] = self.data["time"].dt.hour
        self.data["minute"] = self.data["time"].dt.minute

        # 将时间转换为循环特征
        self.data["hour_sin"] = np.sin(2 * np.pi * self.data["hour"] / 24)
        self.data["hour_cos"] = np.cos(2 * np.pi * self.data["hour"] / 24)
        self.data["minute_sin"] = np.sin(2 * np.pi * self.data["minute"] / 60)
        self.data["minute_cos"] = np.cos(2 * np.pi * self.data["minute"] / 60)

        # 对心率数据进行归一化
        self.data["heart_rate"] = (self.data["heart_rate"] - self.data["heart_rate"].mean()) / self.data[
            "heart_rate"
        ].std()

        # 将睡眠分期转换为数值标签
        stage_mapping = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
        self.data["sleep_stage"] = self.data["sleep_stage"].map(stage_mapping)

        # 准备特征和标签
        self.features = self.data[["heart_rate", "hour_sin", "hour_cos", "minute_sin", "minute_cos"]].values
        self.labels = self.data["sleep_stage"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        return features, label
