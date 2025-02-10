import re
import xml.etree.ElementTree as ET

import mne
import numpy as np
import heartpy as hp
from scipy.signal import detrend, find_peaks
import matplotlib.pyplot as plt


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

    def __init__(self, edf_file_path: str, interval: float = None):
        print(f"初始化 EDFExtractor: 文件路径={edf_file_path}, 采样间隔={interval}秒")
        self.edf_file_path = edf_file_path
        self._edf = None
        self.interval = interval

    def _load_edf(self):
        """加载EDF文件"""
        if self._edf is None:
            # 加载文件
            print("加载EDF文件...")
            self._edf = mne.io.read_raw_edf(
                self.edf_file_path, preload=True
            )  # 改为 preload=True
            original_sfreq = self._edf.info["sfreq"]
            print(f"原始采样率: {original_sfreq} Hz")

            if self.interval is None:
                print("未设置采样周期，无需重采样")

            # 计算目标采样率
            target_sfreq = 1.0 / self.interval
            print(f"目标采样率: {target_sfreq} Hz (基于间隔 {self.interval}秒)")

            # 确保目标采样率不高于原始采样率
            if target_sfreq > original_sfreq:
                print(
                    f"警告：请求的采样间隔（{self.interval}秒）小于数据本身的最小采样间隔（{1.0 / original_sfreq}秒）"
                )
                print(f"将使用最小可能的采样间隔：{1.0 / original_sfreq}秒")
                target_sfreq = original_sfreq

            # 执行重采样
            if self._edf.info["sfreq"] != target_sfreq:
                print(f"执行重采样: {self._edf.info['sfreq']} Hz -> {target_sfreq} Hz")
                self._edf.resample(target_sfreq)
                print("重采样完成")
            else:
                print("无需重采样，当前采样率已符合要求")

    def get_channel_data(self, channel: str, raw: bool = False) -> tuple:
        """
        获取指定通道的数据

        Args:
            channel (str): 通道名称
            raw (bool): 是否返回原始采样率的数据，默认False

        Returns:
            tuple: (数据数组, 时间戳数组)
        """
        print(f"开始获取通道[{channel}]数据")

        if raw:
            # 对于原始数据，直接加载不进行重采样
            if self._edf is None:
                print("加载EDF文件...")
                self._edf = mne.io.read_raw_edf(self.edf_file_path, preload=True)
        else:
            # 对于非原始数据，使用常规的加载过程
            self._load_edf()

        all_channels = self._edf.ch_names
        print(f"可用通道列表: {all_channels}")

        matched_channels = [
            ch for ch in all_channels if re.search(channel, ch, re.IGNORECASE)
        ]
        print(f"匹配到的通道: {matched_channels}")

        if not matched_channels:
            raise ValueError(f"No channels found matching pattern: {channel}")

        channel = matched_channels[0]
        print(f"使用通道: {channel}")

        # 获取数据
        data, times = self._edf[channel]
        data = data.flatten()

        if raw:
            print(f"返回原始采样率数据，采样率: {self._edf.info['sfreq']} Hz")
            return data, times

        # 计算降采样的步长
        original_interval = 1.0 / self._edf.info["sfreq"]
        step = max(1, int(self.interval / original_interval))

        # 手动降采样
        data = data[:-1:step]
        times = times[:-1:step]

        return data, times

    def ecg_to_hr(self, ecg_data: np.ndarray, ecg_timestamps: np.ndarray) -> tuple:
        """
        将ECG数据转换为心率数据

        Args:
            ecg_data (np.ndarray): 原始ECG数据
            ecg_timestamps (np.ndarray): ECG数据对应的时间戳

        Returns:
            tuple: (心率数据数组, 30s间隔的时间戳数组)
        """
        print("开始处理ECG数据...")

        # 计算采样率
        sampling_rate = 1.0 / (ecg_timestamps[1] - ecg_timestamps[0])
        print(f"ECG数据采样率: {sampling_rate} Hz")

        if sampling_rate < 100:  # 如果采样率低于100Hz，可能无法准确检测R波
            raise ValueError(
                f"ECG采样率太低 ({sampling_rate} Hz)，需要原始采样率的数据。请使用raw=True获取数据。"
            )

        # 数据预处理
        # 1. 移除基线漂移
        ecg_detrended = detrend(ecg_data)

        # 2. 标准化数据
        ecg_normalized = (ecg_detrended - np.mean(ecg_detrended)) / np.std(
            ecg_detrended
        )

        # 使用find_peaks检测R波峰
        # 确保最小距离至少为1
        min_distance = max(1, int(sampling_rate * 0.5))  # 最小RR间隔0.5秒

        # 设置更稳健的峰值检测参数
        peaks, _ = find_peaks(
            ecg_normalized,
            distance=min_distance,
            height=0.5,  # 标准化后的相对高度阈值
            prominence=0.5,
        )  # 要求峰值明显突出

        print(f"检测到 {len(peaks)} 个R波峰")

        if len(peaks) < 2:
            raise ValueError("检测到的R波峰数量过少，无法计算心率")

        # 计算RR间隔（以秒为单位）
        rr_intervals = np.diff(ecg_timestamps[peaks])
        rr_times = ecg_timestamps[peaks[1:]]

        # 将RR间隔转换为瞬时心率（次/分钟）
        instant_hr = 60.0 / rr_intervals

        # 移除异常值（例如，心率<20或>200的值）
        valid_mask = (instant_hr >= 20) & (instant_hr <= 200)
        instant_hr = instant_hr[valid_mask]
        hr_times = rr_times[valid_mask]

        print(f"有效心率数据点数: {len(instant_hr)}")

        if len(instant_hr) == 0:
            raise ValueError("没有有效的心率数据点")

        # 创建30秒间隔的时间窗口
        start_time = ecg_timestamps[0]
        end_time = ecg_timestamps[-1]
        window_times = np.arange(start_time, end_time, self.interval)

        # 计算每个30秒窗口的平均心率
        heart_rates = []
        for i in range(len(window_times) - 1):
            window_start = window_times[i]
            window_end = window_times[i + 1]

            # 找到当前时间窗口内的心率值
            mask = (hr_times >= window_start) & (hr_times < window_end)
            window_hr = instant_hr[mask]

            # 如果窗口内有数据，计算平均值；否则使用插值或前一个值
            if len(window_hr) > 0:
                heart_rates.append(np.mean(window_hr))
            else:
                # 如果是第一个窗口且没有数据，使用后面的有效值
                if i == 0 and len(heart_rates) == 0:
                    next_valid = (
                        instant_hr[hr_times >= window_end][0]
                        if any(hr_times >= window_end)
                        else 60
                    )
                    heart_rates.append(next_valid)
                # 否则使用前一个窗口的值
                else:
                    heart_rates.append(heart_rates[-1])

        print(f"生成了 {len(heart_rates)} 个30秒间隔的心率数据点")

        return np.array(heart_rates), window_times[:-1]

    def parse_ecg(
        self, ecg_data: np.ndarray, ecg_timestamps: np.ndarray, window_size: int = 30
    ) -> tuple:
        print("开始处理ECG数据以计算HRV...")

        # 计算采样率
        sampling_rate = 1.0 / (ecg_timestamps[1] - ecg_timestamps[0])
        print(f"ECG数据采样率: {sampling_rate} Hz")

        if sampling_rate < 100:  # 如果采样率低于100Hz，可能无法准确检测R波
            raise ValueError(
                f"ECG采样率太低 ({sampling_rate} Hz)，需要原始采样率的数据。请使用raw=True获取数据。"
            )

        ecg_filtered = hp.filter_signal(
            ecg_data,
            cutoff=[0.8, 2.5],
            sample_rate=sampling_rate,
            filtertype="bandpass",
        )

        max_time = int(ecg_timestamps[-1])
        n_windows = int(np.floor(max_time / window_size))

        hr_list = []
        hrv_list = []
        hr_window_size = int(sampling_rate * 3)

        for i in range(n_windows):
            start_time = i * window_size
            end_time = start_time + window_size

            start_index = int(start_time * sampling_rate)
            end_index = int(end_time * sampling_rate)

            ecg_segment_hrv = ecg_filtered[start_index:end_index]

            try:
                _, measures = hp.process(
                    ecg_segment_hrv,
                    sample_rate=sampling_rate,
                    calc_freq=True,
                    freq_method="welch",
                    interp_clipping=True,
                    reject_segmentwise=True,
                )
                hrv_list.append(measures["rmssd"])
            except Exception as e:
                print(f"计算第{i+1}个HRV窗口时出现错误: {str(e)}")
                hrv_list.append(-1)

            ecg_segment_hr = (
                ecg_segment_hrv[hr_window_size * -1 :]
                if len(ecg_segment_hrv) > hr_window_size
                else ecg_segment_hrv
            )

            try:
                _, measures = hp.process(
                    ecg_segment_hr,
                    sample_rate=sampling_rate,
                    calc_freq=False,
                    interp_clipping=True,
                    reject_segmentwise=True,
                )
                hr_list.append(measures["bpm"])
            except Exception as e:
                print(f"计算第{i+1}个HR窗口时出现错误: {str(e)}")
                hr_list.append(-1)


        # 处理最后一个窗口
        final_ecg_segment_hrv = ecg_filtered[
            int(n_windows * window_size * sampling_rate) :
        ]
        
        try:
            _, measures = hp.process(
                final_ecg_segment_hrv,
                sample_rate=sampling_rate,
                calc_freq=True,
                freq_method="welch",
                interp_clipping=True,
                reject_segmentwise=True,
            )
            hrv_list.append(measures["rmssd"])
        except Exception as e:
            print(f"计算最后一个HRV窗口时出现错误: {str(e)}")
            hrv_list.append(-1)

        final_ecg_segment_hr = (
            final_ecg_segment_hrv[hr_window_size * -1 :]
            if len(final_ecg_segment_hrv) > hr_window_size
            else final_ecg_segment_hrv
        )
        
        try:
            _, measures = hp.process(
                final_ecg_segment_hr,
                sample_rate=sampling_rate,
                calc_freq=False,
                interp_clipping=True,
                reject_segmentwise=True,
            )
            hr_list.append(measures["bpm"])
        except Exception as e:
            print(f"计算最后一个HR窗口时出现错误: {str(e)}")
            hr_list.append(-1)

        # 进一步剔除异常值
        hr_list = [round(hr) if hr > 40 and hr < 200 else -1 for hr in hr_list]
        hrv_list = [round(hrv) if hrv > 0 and hrv < 160 else -1 for hrv in hrv_list]

        hr_num = len(hr_list)
        hrv_num = len(hrv_list)

        if hr_num != hrv_num:
            print(
                f"心率数据点数({hr_num})与RMSSD数据点数({hrv_num})不一致，将使用较小的数据点数"
            )
            n_windows = min(hr_num, hrv_num)
            hr_list = hr_list[:n_windows]
            hrv_list = hrv_list[:n_windows]
        else:
            print(f"提取了 {hr_num} 个{window_size}秒间隔的数据点")

        return (
            np.array(hr_list),
            np.array(hrv_list),
            np.arange(0, n_windows * window_size, window_size),
        )
