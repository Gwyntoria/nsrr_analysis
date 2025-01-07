import re
import xml.etree.ElementTree as ET

import mne


class XMLExtractor:
    """XML数据提取器类"""

    def __init__(self, xml_file_path, element_name):
        """
        初始化XML提取器

        Args:
            xml_file_path (str): XML文件的路径
        """
        self.xml_file_path = xml_file_path
        self.element_name = element_name

    def extract_sleep_stages(self) -> list:
        """
        从XML文件中提取SleepStage的值

        Returns:
            list: 包含所有SleepStage值的列表

        Raises:
            FileNotFoundError: 当XML文件不存在时
            ET.ParseError: 当XML解析出错时
        """
        # 解析XML文件
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()

        sleep_stages = []
        # 使用 .// 进行递归查找所有层级中的元素
        for stage in root.findall(f".//{self.element_name}"):
            sleep_stages.append(int(stage.text))

        return sleep_stages


class EDFExtractor:
    """EDF数据提取器类，用于从EDF文件中提取指定通道的数据和时间戳"""

    def __init__(self, edf_file_path: str, interval: float = 1.0):
        print(f"初始化 EDFExtractor: 文件路径={edf_file_path}, 采样间隔={interval}秒")
        self.edf_file_path = edf_file_path
        self._edf = None
        self.chunk_size = 1000000
        self.interval = interval

    def _load_edf(self):
        """加载EDF文件"""
        if self._edf is None:
            # 加载文件
            print("加载EDF文件...")
            self._edf = mne.io.read_raw_edf(self.edf_file_path, preload=True)  # 改为 preload=True
            original_sfreq = self._edf.info['sfreq']
            print(f"原始采样率: {original_sfreq} Hz")
            
            # 计算目标采样率
            target_sfreq = 1.0 / self.interval
            print(f"目标采样率: {target_sfreq} Hz (基于间隔 {self.interval}秒)")
            
            # 确保目标采样率不高于原始采样率
            if target_sfreq > original_sfreq:
                print(f"警告：请求的采样间隔（{self.interval}秒）小于数据本身的最小采样间隔（{1.0/original_sfreq}秒）")
                print(f"将使用最小可能的采样间隔：{1.0/original_sfreq}秒")
                target_sfreq = original_sfreq
            
            # 执行重采样
            if self._edf.info['sfreq'] != target_sfreq:
                print(f"执行重采样: {self._edf.info['sfreq']} Hz -> {target_sfreq} Hz")
                self._edf.resample(target_sfreq)
                print("重采样完成")
            else:
                print("无需重采样，当前采样率已符合要求")

    def get_channel_data(self, channel: str) -> tuple:
        print(f"开始获取通道[{channel}]数据")
        self._load_edf()

        all_channels = self._edf.ch_names
        print(f"可用通道列表: {all_channels}")
        
        matched_channels = [ch for ch in all_channels if re.search(channel, ch, re.IGNORECASE)]
        print(f"匹配到的通道: {matched_channels}")

        if not matched_channels:
            raise ValueError(f"No channels found matching pattern: {channel}")

        channel = matched_channels[0]
        print(f"使用通道: {channel}")

        # 获取原始数据
        data, times = self._edf[channel]
        data = data.flatten()
        
        # 计算降采样的步长
        original_interval = 1.0 / self._edf.info['sfreq']
        step = max(1, int(self.interval / original_interval))
        
        # 手动降采样
        data = data[:-1:step]
        times = times[:-1:step]
        
        return data, times
