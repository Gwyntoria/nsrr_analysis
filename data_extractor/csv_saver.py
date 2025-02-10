import os

import pandas as pd


class CSVSaver:
    """CSV数据保存器类"""

    # 类变量，用于存储实例缓存
    _instance_cache = {}

    def __init__(self, output_file):
        """
        初始化CSV保存器

        Args:
            output_file (str): 输出CSV文件的路径
        """
        self.output_file = output_file

    def __new__(cls, output_file):
        """
        确保同一文件路径只创建一个实例

        Args:
            output_file (str): 输出CSV文件的路径

        Returns:
            CSVSaver: CSVSaver实例
        """
        if output_file not in cls._instance_cache:
            cls._instance_cache[output_file] = super().__new__(cls)
        return cls._instance_cache[output_file]

    def save_csv_element(self, column_name, column_values):
        """
        将数据保存到CSV文件，新数据会添加为新的一列

        Args:
            column_name (str): 新列的名称
            column_values (list): 值的列表

        Raises:
            OSError: 当创建目录或写入文件失败时
        """
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        try:
            # 尝试读取现有文件
            existing_df = pd.read_csv(self.output_file)

        except (FileNotFoundError, pd.errors.EmptyDataError):
            # 如果文件不存在或为空，创建新的DataFrame
            existing_df = pd.DataFrame({"Index": range(1, len(column_values) + 1)})

        # 添加新的数据列
        existing_df[column_name] = pd.Series(column_values)

        # 保存到CSV文件
        existing_df.to_csv(self.output_file, index=False)
