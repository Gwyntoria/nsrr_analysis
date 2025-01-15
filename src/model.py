import torch.nn as nn


class SleepStageClassifier(nn.Module):
    def __init__(self, hidden_size=64, num_classes=5):
        """
        睡眠分期分类器
        Args:
            hidden_size: 隐藏层大小
            num_classes: 分类数量（睡眠分期数）
        """
        super(SleepStageClassifier, self).__init__()

        # 输入特征维度为5: [heart_rate, hour_sin, hour_cos, minute_sin, minute_cos]
        input_size = 5

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)
