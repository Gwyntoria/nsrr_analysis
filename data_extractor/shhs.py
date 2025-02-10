import os
import re
import time
import traceback

from csv_saver import CSVSaver
from extractor import EDFExtractor, XMLExtractor

edf_dir = "E:\\Dataset\\nsrr\\shhs\\polysomnography\\edfs"
xml_dir = "E:\\Dataset\\nsrr\\shhs\\polysomnography\\annotations-events-profusion"
csv_dir = "E:\\Dataset\\nsrr\\shhs\\polysomnography\\csvs-2"


def get_sorted_files(directory, extension) -> list:
    """
    递归获取目录及其子目录下指定扩展名的文件并按数字排序

    Args:
        directory (str): 目录路径
        extension (str): 文件扩展名

    Returns:
        list: 包含(子目录路径, 文件名)元组的排序列表
    """
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                # 保存(子目录路径, 文件名)元组
                result.append((root, f))
    # 按文件名中的数字排序
    return sorted(result, key=lambda x: int(re.search(r"\d+", x[1]).group()))


def find_matching_xml(edf_filename, xml_files) -> tuple:
    """
    根据edf文件名找到对应的xml文件

    Args:
        edf_filename (str): edf文件名
        xml_files (list): 包含(路径,文件名)元组的xml文件列表

    Returns:
        tuple: 匹配的xml文件的(路径,文件名)元组，未找到则返回(None, None)
    """
    # 从edf文件名中提取编号部分
    edf_base = os.path.splitext(edf_filename)[0]  # 移除扩展名
    xml_pattern = f"{edf_base}-profusion.xml"

    for xml_path, xml_file in xml_files:
        if xml_file == xml_pattern:
            return (xml_path, xml_file)
    return (None, None)


if __name__ == "__main__":
    try:
        print("开始处理数据...")
        # 获取排序后的edf文件列表
        edf_files = get_sorted_files(edf_dir, ".edf")
        xml_files = get_sorted_files(xml_dir, ".xml")

        edf_files_num = len(edf_files)
        xml_files_num = len(xml_files)
        print(f"找到 {edf_files_num} 个EDF文件和 {xml_files_num} 个XML文件")

        if edf_files_num != xml_files_num:
            raise ValueError("EDF文件和XML文件数量不匹配")

        file_counter = 0
        error_files = []  # 用于存储处理出错的文件

        for edf_path, edf_file in edf_files:
            try:
                file_counter += 1
                progress = (file_counter / edf_files_num) * 100
                print(
                    f"处理第{file_counter}/{edf_files_num}个文件 ({progress:.2f}%): {edf_file}"
                )

                # 找到对应的xml文件
                xml_path, xml_file = find_matching_xml(edf_file, xml_files)
                if not xml_file:
                    print(f"警告：未找到与 {edf_file} 对应的XML文件")
                    continue

                full_edf_path = os.path.join(edf_path, edf_file)
                full_xml_path = os.path.join(xml_path, xml_file)
                print(f"对应的XML文件: {xml_file}")

                # 创建与EDF文件相同的子目录结构
                relative_path = os.path.relpath(edf_path, edf_dir)
                output_dir = os.path.join(csv_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # 提取文件编号用于csv文件命名
                file_base = os.path.splitext(edf_file)[0]
                csv_path = os.path.join(output_dir, f"{file_base}.csv")
                print(f"输出CSV文件: {csv_path}")

                # 检查CSV文件是否已存在
                if os.path.exists(csv_path):
                    print(f"CSV文件 {csv_path} 已存在，跳过处理")
                    print("=" * 50)
                    continue

                print("创建数据提取器...")
                edf_extractor = EDFExtractor(full_edf_path, interval=30)
                xml_extractor = XMLExtractor(full_xml_path, "SleepStage")

                print("开始提取数据...")
                # 获取原始采样率的ECG数据
                ecg_data, ecg_timestamps = edf_extractor.get_channel_data(
                    "ECG", raw=True
                )
                print(f"ECG数据点数: {len(ecg_data)}")

                hr_list, hrv_list, time_list = edf_extractor.parse_ecg(
                    ecg_data, ecg_timestamps, window_size=30
                )
                hr_num = len(hr_list)
                print(f"心率数据点数: {hr_num}")

                sleep_stage_list = xml_extractor.extract_sleep_stages()
                sleep_stage_num = len(sleep_stage_list)
                print(f"睡眠阶段数据点数: {sleep_stage_num}")

                if hr_num != sleep_stage_num:
                    print(f"心率数据点数与睡眠阶段数据点数不匹配,裁剪数据")
                    min_num = min(hr_num, sleep_stage_num)
                    hr_list = hr_list[:min_num]
                    hrv_list = hrv_list[:min_num]
                    time_list = time_list[:min_num]
                    sleep_stage_list = sleep_stage_list[:min_num]

                print("保存数据到CSV...")
                saver = CSVSaver(csv_path)
                saver.save_csv_element("timestamp", time_list)
                saver.save_csv_element("HR", hr_list)
                saver.save_csv_element("HRV", hrv_list)
                saver.save_csv_element("sleep_stage", sleep_stage_list)

                print(f"成功处理文件 {edf_file}")
                print(f"数据已保存到: {csv_path}")

                print("=" * 50)
                # 添加系统延时10ms
                time.sleep(0.01)

            except ValueError as ve:
                print(f"\n处理文件 {edf_file} 时出现错误: {str(ve)}")
                print("跳过当前文件，继续处理下一个文件...")
                error_files.append((edf_file, str(ve)))  # 记录错误文件和错误信息
                continue
            except Exception as e:
                print(f"\n处理文件 {edf_file} 时出现未预期的错误: {str(e)}")
                print("详细错误信息:")
                print(traceback.format_exc())
                print("跳过当前文件，继续处理下一个文件...")
                error_files.append((edf_file, str(e)))  # 记录错误文件和错误信息
                continue

        # 处理完所有文件后，打印错误文件列表
        if error_files:
            print("\n以下文件在处理过程中出现错误：")
            for file_name, error_msg in error_files:
                print(f"文件: {file_name}")
                print(f"错误信息: {error_msg}")
                print("-" * 50)

            # 将错误文件信息保存到error-file.txt
            error_file_path = os.path.join(csv_dir, "error-file.txt")
            try:
                with open(error_file_path, "w", encoding="utf-8") as f:
                    f.write("处理失败的文件列表：\n")
                    f.write("=" * 50 + "\n")
                    for file_name, error_msg in error_files:
                        f.write(f"文件: {file_name}\n")
                        f.write(f"错误信息: {error_msg}\n")
                        f.write("-" * 50 + "\n")
                print(f"\n错误文件列表已保存到: {error_file_path}")
            except Exception as e:
                print(f"\n保存错误文件列表时出现错误: {str(e)}")
        else:
            print("\n所有文件处理完成，没有出现错误！")

    except Exception as e:
        print(f"\n程序执行过程中出现错误: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())
