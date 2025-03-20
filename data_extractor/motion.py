import os
import pandas as pd
import glob

HEART_RATE_DIR = "E:\\Dataset\\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0\\heart_rate"
LABEL_DIR = "E:\\Dataset\\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0\\labels"
MOTION_DIR = "E:\\Dataset\\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0\\motion"
STEP_DIR = "E:\\Dataset\\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0\\steps"
CSV_DIR = "E:\\Dataset\\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0\\csv_acc"


def read_data_file(file_path):
    """读取数据文件并过滤负时间戳"""
    try:
        # 检查文件名以确定分隔符
        if "_labeled_sleep.txt" in file_path:
            # 标签文件使用空格分隔
            df = pd.read_csv(
                file_path, header=None, names=["timestamp", "sleep stage"], sep=r"\s+"
            )
        elif "_acceleration.txt" in file_path:
            # 运动文件使用空格分隔
            df = pd.read_csv(
                file_path,
                header=None,
                names=["timestamp", "acc_x", "acc_y", "acc_z"],
                sep=r"\s+",
            )
        else:
            # 其他文件使用逗号分隔
            df = pd.read_csv(file_path, header=None, names=["timestamp", "value"])
            # 去除timestamp列中的空格
            if df["timestamp"].dtype == object:  # 如果是字符串类型
                df["timestamp"] = df["timestamp"].str.strip()

        # 将timestamp转换为数值类型
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        # 过滤掉负时间戳的数据和无效数据
        df = df.dropna()
        df = df[df["timestamp"] >= 0]
        # 确保timestamp是整数类型
        df["timestamp"] = df["timestamp"].astype(int)
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None


def process_subject_data(subject_id):
    """处理单个受试者的所有数据"""
    data = {}

    # 读取心率数据
    heart_rate_file = os.path.join(HEART_RATE_DIR, f"{subject_id}_heartrate.txt")
    if os.path.exists(heart_rate_file):
        data["heart_rate"] = read_data_file(heart_rate_file)

    # 读取睡眠标签数据
    label_file = os.path.join(LABEL_DIR, f"{subject_id}_labeled_sleep.txt")
    if os.path.exists(label_file):
        data["label"] = read_data_file(label_file)

    # 读取运动数据
    motion_file = os.path.join(MOTION_DIR, f"{subject_id}_acceleration.txt")
    if os.path.exists(motion_file):
        data["motion"] = read_data_file(motion_file)

    # # 读取步数数据
    # step_file = os.path.join(step_dir, f"{subject_id}_steps.txt")
    # if os.path.exists(step_file):
    #     data["steps"] = read_data_file(step_file)

    return data


def save_to_csv(subject_id, data):
    """将处理后的数据保存为CSV文件"""
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)

    output_file = os.path.join(CSV_DIR, f"{subject_id}.csv")

    # 合并所有数据
    result = pd.DataFrame()
    for data_type, df in data.items():
        if df is not None and not df.empty:
            # 确保timestamp是整数类型
            df["timestamp"] = df["timestamp"].astype(int)
            # 重命名value列以区分不同类型的数据
            df = df.rename(columns={"value": data_type})
            if result.empty:
                result = df
            else:
                # 基于时间戳合并数据
                result = pd.merge(result, df, on="timestamp", how="outer")

    if not result.empty:
        # 按时间戳排序
        result = result.sort_values("timestamp")

        # 填充心率数据的缺失值
        if "heart_rate" in result.columns:
            # 使用前向填充方法填充心率数据的空缺值
            result["heart_rate"] = result["heart_rate"].bfill()

        # 将sleep stage列转换为整数类型，并删除空值所在的行
        if "sleep stage" in result.columns:
            result["sleep stage"] = pd.to_numeric(
                result["sleep stage"], errors="coerce"
            )
            result = result.dropna(subset=["sleep stage"])
            result["sleep stage"] = result["sleep stage"].astype(int)

        result.to_csv(output_file, index=False)
        print(f"Saved data for subject {subject_id}")


def main():
    """主函数"""
    # 获取所有心率文件的ID
    heart_rate_files = glob.glob(os.path.join(HEART_RATE_DIR, "*_heartrate.txt"))
    for file_path in heart_rate_files:
        # 从文件名中提取subject_id
        subject_id = os.path.basename(file_path).split("_")[0]

        # 处理该受试者的所有数据
        data = process_subject_data(subject_id)

        # 保存处理后的数据
        save_to_csv(subject_id, data)


if __name__ == "__main__":
    main()
