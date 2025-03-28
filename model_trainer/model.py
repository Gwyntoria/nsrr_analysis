import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PathConfig, LOG_LEVEL, setup_logger

# 配置日志
logger = setup_logger(
    name=__name__,
    log_file=os.path.join(PathConfig.logs_dir, "training.log"),
    level=LOG_LEVEL,
)


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # 加权求和
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class SleepStageClassifier(nn.Module):
    def __init__(
        self,
        input_size=4,
        hidden_size=256,
        num_layers=3,
        num_classes=4,
        dropout=0.5,
    ):
        """
        基于LSTM的睡眠分期分类器
        Args:
            input_size: 输入特征维度 (4个特征: relative_time, heart_rate, heart_rate_diff,
                       heart_rate_ma)
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            num_classes: 分类数量（4个睡眠分期）
            dropout: dropout比率
        """
        super(SleepStageClassifier, self).__init__()

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # 注意力层
        self.attention = AttentionLayer(hidden_size)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # 添加状态转移预测层
        self.transition_predictor = nn.Linear(
            hidden_size * 2, num_classes * num_classes
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 特征提取
        x = self.feature_extractor(x)

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 注意力机制
        context_vector, attention_weights = self.attention(lstm_out)

        # 主要分类预测
        main_output = self.classifier(context_vector)

        # 状态转移预测
        transition_logits = self.transition_predictor(context_vector)
        transition_matrix = transition_logits.view(batch_size, 4, 4)

        return main_output, transition_matrix, attention_weights
