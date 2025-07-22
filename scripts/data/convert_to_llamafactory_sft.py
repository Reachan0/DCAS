import pandas as pd
import json
from tqdm import tqdm

# --- 1. 配置区域 ---

# 输入的CSV文件路径 (包含教师模型生成的skill_requirements列)
INPUT_CSV_PATH = 'datasets/job_skills_dataset_20250712_132024.csv'

# 最终输出的Alpaca格式JSONL文件路径
OUTPUT_JSONL_PATH = 'datasets/job_skills_alpaca_finetune.jsonl'

# Alpaca格式的"指令" (instruction)，这是固定的
INSTRUCTION_TEXT = "根据以下职位名称和描述，提取并总结出最核心的能力要求。请以一个由英文逗号和单个空格分隔的字符串形式返回。"


# --- 2. 主执行逻辑 ---

def convert_to_alpaca(input_path, output_path, instruction):
    """
    读取CSV，清洗数据，并将其转换为包含 'instruction', 'input', 'output' 的Alpaca JSONL格式。
    """
    try:
        # --- 数据加载与清洗 ---
        print(f"开始从 '{input_path}' 加载数据...")
        df = pd.read_csv(input_path)
        original_count = len(df)
        print(f"加载了 {original_count} 条原始数据。")

        # 确保用于Alpaca格式的三个核心列都存在且不为空
        # 'job_title' 和 'job_description' 用于构建 'input'
        # 'skill_requirements' 用于构建 'output'
        required_cols = ['job_title', 'job_description', 'skill_requirements']

        # 丢弃任何核心列为空的行
        df.dropna(subset=required_cols, inplace=True)
        # 将空字符串也视为空值并丢弃
        for col in required_cols:
            df = df[df[col].astype(str).str.strip() != '']

        df.reset_index(drop=True, inplace=True)
        cleaned_count = len(df)
        print(f"数据清洗完成，移除了 {original_count - cleaned_count} 条无效行，剩余 {cleaned_count} 条有效数据。")
        # --- 清洗结束 ---

        # --- 格式转换 ---
        with open(output_path, 'w', encoding='utf-8') as f:
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="正在转换为Alpaca格式"):
                # 构造 "input" 字段的内容
                input_content = f"【职位名称】:\n{row['job_title']}\n\n【职位描述】:\n{row['job_description']}"

                # 构造 "output" 字段的内容
                output_content = str(row['skill_requirements'])

                # 组装成完整的Alpaca格式字典
                alpaca_entry = {
                    "instruction": instruction,
                    "input": input_content,
                    "output": output_content  # <--- 确保这一步将skill_requirements赋给了output
                }

                # 将字典转换为JSON字符串并写入文件
                f.write(json.dumps(alpaca_entry, ensure_ascii=False) + '\n')

        print(f"\n转换成功！Alpaca格式的数据集已保存至: '{output_path}'")
        print("请检查文件中的任意一行，确认其包含 'instruction', 'input', 和 'output' 三个字段。")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_path}'。请检查路径是否正确。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")


# --- 3. 运行脚本 ---
if __name__ == "__main__":
    convert_to_alpaca(INPUT_CSV_PATH, OUTPUT_JSONL_PATH, INSTRUCTION_TEXT)