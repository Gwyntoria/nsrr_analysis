import os
import pandas as pd
import time


from extractor import EDFExtractor
from csv_saver import CSVSaver
from shhs import get_sorted_files

edf_dir = "E:\\Dataset\\nsrr\\shhs\\polysomnography\\edfs"
xml_dir = "E:\\Dataset\\nsrr\\shhs\\polysomnography\\annotations-events-profusion"
hrv_dir = "E:\\Dataset\\nsrr\\shhs\\polysomnography\\hrvs"


def main():
    edf_files = get_sorted_files(edf_dir, ".edf")
    edf_files_num = len(edf_files)
    print(f"找到 {edf_files_num} 个EDF文件")

    file_counter = 0
    error_files = []

    for edf_path, edf_file in edf_files:
        try:
            file_counter += 1
            progress = (file_counter / edf_files_num) * 100
            print(
                f"处理第{file_counter}/{edf_files_num}个文件 ({progress:.2f}%): {edf_file}"
            )

            full_edf_path = os.path.join(edf_path, edf_file)
            print(f"处理文件: {edf_file}")

            # 创建与CSV文件相同的子目录结构
            relative_path = os.path.relpath(edf_path, edf_dir)
            output_dir = os.path.join(hrv_dir, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            hrv_csv_path = os.path.join(
                output_dir, f"{os.path.splitext(edf_file)[0]}_hrv.csv"
            )
            if os.path.exists(hrv_csv_path):
                print(f"HRV 文件 {hrv_csv_path} 已存在，跳过处理")
                print("=" * 50)
                continue

            print("创建数据提取器...")
            edf_extractor = EDFExtractor(full_edf_path, interval=1 / 75)
            ecg_data, ecg_timestamps = edf_extractor.get_channel_data("ECG", raw=True)

            print("开始提取HRV数据...")
            hr_values, hrv_values, timestamps = edf_extractor.parse_ecg(ecg_data, ecg_timestamps)

            print("保存数据到CSV...")
            saver = CSVSaver(hrv_csv_path)
            saver.save_csv_element("timestamp", timestamps)
            saver.save_csv_element("hr", hr_values)
            saver.save_csv_element("hrv", hrv_values)

            print(f"HRV数据已保存到: {hrv_csv_path}")

            print("=" * 50)
            time.sleep(0.01)

        except ValueError as ve:
            print(f"\n处理文件 {edf_file} 时出现错误: {str(ve)}")
            print("跳过当前文件，继续处理下一个文件...")
            error_files.append((edf_file, str(ve)))  # 记录错误文件和错误信息
            continue

        except Exception as e:
            print(f"处理文件 {edf_file} 时出现错误: {str(e)}")
            error_files.append((edf_file, str(e)))

    if error_files:
        print("\n以下文件在处理过程中出现错误：")
        for file_name, error_msg in error_files:
            print(f"文件: {file_name}, 错误信息: {error_msg}")

    print("\n所有HRV数据提取完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n========== 用户中断了程序执行。 ==========")
