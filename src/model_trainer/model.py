import torch.nn as nn


class SleepStageClassifier(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, num_classes=6, dropout=0.3):
        """
        基于LSTM的睡眠分期分类器
        Args:
            input_size: 输入特征维度 (7个特征: heart_rate, heart_rate_diff, heart_rate_diff2,
                       hour_sin, hour_cos, minute_sin, minute_cos)
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            num_classes: 分类数量（6个睡眠分期）
            dropout: dropout比率
        """
        super(SleepStageClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        # 双向LSTM，所以hidden_size要乘2
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # LSTM forward
        lstm_out, _ = self.lstm(x)
        # 只使用序列的最后一个输出
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
