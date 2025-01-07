import os
import re
import traceback

from csv_saver import CSVSaver
from extractor import EDFExtractor, XMLExtractor

edf_dir = "E:\\Dataset\\nsrr\\mesa\\polysomnography\\edfs"
xml_dir = "E:\\Dataset\\nsrr\\mesa\\polysomnography\\annotations-events-profusion"
csv_dir = "E:\\Dataset\\nsrr\\mesa\\polysomnography\\csvs"


def get_sorted_files(directory, extension) -> list:
    """
    获取目录下指定扩展名的文件并按数字排序

    Args:
        directory (str): 目录路径
        extension (str): 文件扩展名

    Returns:
        list: 排序后的文件路径列表
    """
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    # 提取文件名中的数字并排序
    return sorted(files, key=lambda x: int(re.search(r"\d+", x).group()))


def find_matching_xml(edf_filename, xml_files) -> str:
    """
    根据edf文件名找到对应的xml文件

    Args:
        edf_filename (str): edf文件名
        xml_files (list): xml文件列表

    Returns:
        str: 匹配的xml文件名，未找到则返回None
    """
    edf_number = re.search(r"\d+", edf_filename).group()
    for xml_file in xml_files:
        if edf_number in xml_file:
            return xml_file
    return None


if __name__ == "__main__":
    try:
        print("\n开始处理数据...")
        # 获取排序后的edf文件列表
        edf_files = get_sorted_files(edf_dir, ".edf")
        xml_files = get_sorted_files(xml_dir, ".xml")
        
        edf_files_num = len(edf_files)
        xml_files_num = len(xml_files)
        print(f"找到 {edf_files_num} 个EDF文件和 {edf_files_num} 个XML文件")
        
        if edf_files_num != xml_files_num:
            raise ValueError("EDF文件和XML文件数量不匹配")

        file_counter = 0
        
        for edf_file in edf_files:
            file_counter += 1
            print(f"\n处理第{file_counter}个文件: {edf_file}")
            # 找到对应的xml文件
            xml_file = find_matching_xml(edf_file, xml_files)
            if not xml_file:
                print(f"警告：未找到与 {edf_file} 对应的XML文件")
                continue

            edf_path = os.path.join(edf_dir, edf_file)
            xml_path = os.path.join(xml_dir, xml_file)
            print(f"对应的XML文件: {xml_file}")

            # 提取文件编号用于csv文件命名
            file_number = re.search(r"\d+", edf_file).group()
            csv_path = os.path.join(csv_dir, f"mesa-sleep-{file_number}.csv")
            print(f"输出CSV文件: {csv_path}")

            print("\n创建数据提取器...")
            edf_extractor = EDFExtractor(edf_path, interval=30)
            xml_extractor = XMLExtractor(xml_path, "SleepStage")

            print("\n开始提取数据...")
            heart_rate, timestamps = edf_extractor.get_channel_data("HR")
            print(f"心率数据点数: {len(heart_rate)}")
            
            sleep_stages = xml_extractor.extract_sleep_stages()
            print(f"睡眠阶段数据点数: {len(sleep_stages)}")

            print("\n保存数据到CSV...")
            saver = CSVSaver(csv_path)
            saver.save_csv_element("timestamp", timestamps)
            saver.save_csv_element("heart_rate", heart_rate)
            saver.save_csv_element("sleep_stage", sleep_stages)

            print(f"成功处理文件 {edf_file}")
            print(f"数据已保存到: {csv_path}")

    except Exception as e:
        print(f"\n处理过程中出现错误: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())
