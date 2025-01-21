import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOG_LEVEL, setup_logger

# 配置日志
logger = setup_logger(name="model", log_file="training.log", level=LOG_LEVEL)


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
    def __init__(self, input_size=7, hidden_size=256, num_layers=3, num_classes=6, dropout=0.5):
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

        # 增加输入特征的处理
        self.feature_extractor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.2))

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = AttentionLayer(hidden_size)

        # 增加更深的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # 添加权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # 特征提取
        x = self.feature_extractor(x)

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 注意力机制
        context_vector, _ = self.attention(lstm_out)

        # 分类
        out = self.classifier(context_vector)
        return out
