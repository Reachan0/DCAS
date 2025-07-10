#!/usr/bin/env python3
"""
DCAS - LlamaFactory数据预处理脚本

将CSV格式的职位技能数据转换为LlamaFactory支持的JSONL格式
"""

import pandas as pd
import json
import os
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_llamafactory_format(input_csv: str, output_jsonl: str, dataset_name: str = "job_skills"):
    """
    将CSV数据转换为LlamaFactory格式
    
    Args:
        input_csv: 输入CSV文件路径
        output_jsonl: 输出JSONL文件路径
        dataset_name: 数据集名称
    """
    logger.info(f"开始转换数据: {input_csv} -> {output_jsonl}")
    
    # 读取CSV数据
    df = pd.read_csv(input_csv)
    logger.info(f"加载了 {len(df)} 条记录")
    
    # 转换为LlamaFactory格式
    converted_data = []
    
    for index, row in df.iterrows():
        job_title = row['job_title']
        job_description = row['job_description']
        skill_requirements = row['skill_requirements']
        
        # 构建输入文本
        input_text = f"职位名称: {job_title}\n职位描述: {job_description}\n\n请根据上述职位信息，提取出该职位需要的核心技能要求："
        
        # LlamaFactory标准格式
        # 方案1：指令微调格式
        sample = {
            "instruction": "根据职位信息提取核心技能要求。请只返回技能关键词，用逗号分隔。",
            "input": f"职位名称: {job_title}\n职位描述: {job_description}",
            "output": skill_requirements
        }
        
        converted_data.append(sample)
    
    # 保存为JSONL格式
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"转换完成！生成 {len(converted_data)} 条训练样本")
    logger.info(f"输出文件: {output_jsonl}")
    
    return len(converted_data)

def create_llamafactory_config(dataset_name: str, data_file: str):
    """
    创建LlamaFactory数据集配置
    
    Args:
        dataset_name: 数据集名称
        data_file: 数据文件路径
    """
    config = {
        dataset_name: {
            "file_name": data_file,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input", 
                "response": "output"
            }
        }
    }
    
    config_file = f"llamafactory_dataset_info_{dataset_name}.json"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"生成LlamaFactory配置文件: {config_file}")
    
    return config_file

def create_conversation_format(input_csv: str, output_jsonl: str):
    """
    创建对话格式的训练数据（可选方案）
    
    Args:
        input_csv: 输入CSV文件
        output_jsonl: 输出JSONL文件
    """
    logger.info("创建对话格式数据...")
    
    df = pd.read_csv(input_csv)
    converted_data = []
    
    for index, row in df.iterrows():
        job_title = row['job_title']
        job_description = row['job_description']
        skill_requirements = row['skill_requirements']
        
        # 对话格式
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"请分析以下职位信息并提取核心技能要求：\n\n职位名称: {job_title}\n职位描述: {job_description}\n\n请只返回技能关键词，用逗号分隔。"
                },
                {
                    "from": "gpt", 
                    "value": skill_requirements
                }
            ]
        }
        
        converted_data.append(conversation)
    
    # 保存对话格式
    conversation_file = output_jsonl.replace('.jsonl', '_conversation.jsonl')
    with open(conversation_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"对话格式文件: {conversation_file}")
    return conversation_file

def main():
    """主函数"""
    # 使用清洗后的数据文件
    import glob
    
    # 查找最新的清洗数据文件
    cleaned_files = glob.glob("datasets/cleaned_job_skills_*.csv")
    
    if cleaned_files:
        input_csv = max(cleaned_files)  # 使用最新的清洗文件
        logger.info(f"使用清洗后的数据文件: {input_csv}")
    else:
        # 备选：使用原始temp_results.csv
        input_csv = "datasets/temp_results.csv"
        logger.warning(f"未找到清洗数据，使用原始文件: {input_csv}")
    
    if not os.path.exists(input_csv):
        logger.error(f"未找到数据文件: {input_csv}")
        return
    
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if "cleaned" in input_csv:
        output_jsonl = f"datasets/job_skills_llamafactory_cleaned_{timestamp}.jsonl"
        dataset_name = "dcas_job_skills_cleaned"
    else:
        output_jsonl = f"datasets/job_skills_llamafactory_full_{timestamp}.jsonl"
        dataset_name = "dcas_job_skills_full"
    
    # 转换数据
    sample_count = convert_to_llamafactory_format(input_csv, output_jsonl, dataset_name)
    
    # 创建配置文件
    config_file = create_llamafactory_config(dataset_name, output_jsonl)
    
    # 创建对话格式（备选）
    conversation_file = create_conversation_format(input_csv, output_jsonl)
    
    # 输出使用说明
    print(f"""
=== LlamaFactory微调数据准备完成 ===

数据文件:
- 指令格式: {output_jsonl}
- 对话格式: {conversation_file}
- 配置文件: {config_file}

数据统计:
- 训练样本数: {sample_count}
- 数据质量: {'高质量清洗数据' if 'cleaned' in input_csv else '原始数据'}

使用方法:
1. 将配置文件内容添加到LlamaFactory的dataset_info.json中
2. 使用以下命令开始微调:

llamafactory-cli train \\
    --stage sft \\
    --model_name_or_path qwen \\
    --dataset {dataset_name} \\
    --template qwen \\
    --finetuning_type lora \\
    --output_dir ./saves/qwen-job-skills \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 4 \\
    --lr_scheduler_type cosine \\
    --logging_steps 10 \\
    --save_steps 500 \\
    --learning_rate 5e-5 \\
    --num_train_epochs 3.0 \\
    --plot_loss \\
    --fp16

推荐参数:
- 当前数据集 ({sample_count}条): num_train_epochs=3-5, learning_rate=5e-5
- 如需更好效果: num_train_epochs=5-8, learning_rate=3e-5
""")

if __name__ == "__main__":
    main()