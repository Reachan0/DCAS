#!/usr/bin/env python3
"""
验证LlamaFactory JSONL格式脚本
确保转换后的数据符合LlamaFactory SFT要求
"""

import json
import sys
from typing import Dict, List, Any
from pathlib import Path

def validate_format(jsonl_file: str) -> bool:
    """验证JSONL格式"""
    print(f"开始验证文件: {jsonl_file}")
    
    total_lines = 0
    valid_lines = 0
    errors = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                total_lines += 1
                
                try:
                    data = json.loads(line)
                    
                    # 检查顶层结构
                    if "conversations" not in data:
                        errors.append(f"第{line_num}行: 缺少'conversations'键")
                        continue
                    
                    conversations = data["conversations"]
                    
                    # 检查对话结构
                    if not isinstance(conversations, list):
                        errors.append(f"第{line_num}行: 'conversations'必须是列表")
                        continue
                    
                    if len(conversations) != 2:
                        errors.append(f"第{line_num}行: 对话必须包含2条消息")
                        continue
                    
                    # 检查每条消息的结构
                    for i, msg in enumerate(conversations):
                        if not isinstance(msg, dict):
                            errors.append(f"第{line_num}行: 消息{i+1}必须是字典")
                            continue
                        
                        if "from" not in msg or "value" not in msg:
                            errors.append(f"第{line_num}行: 消息{i+1}缺少'from'或'value'键")
                            continue
                        
                        if msg["from"] not in ["human", "gpt"]:
                            errors.append(f"第{line_num}行: 消息{i+1}的'from'必须是'human'或'gpt'")
                            continue
                        
                        if not isinstance(msg["value"], str) or not msg["value"].strip():
                            errors.append(f"第{line_num}行: 消息{i+1}的'value'必须是非空字符串")
                            continue
                    
                    valid_lines += 1
                    
                except json.JSONDecodeError as e:
                    errors.append(f"第{line_num}行: JSON解析错误 - {e}")
                
    except FileNotFoundError:
        print(f"错误: 文件 {jsonl_file} 不存在")
        return False
    except Exception as e:
        print(f"验证过程出错: {e}")
        return False
    
    # 打印验证结果
    print("="*50)
    print("🔍 格式验证结果")
    print("="*50)
    print(f"总记录数: {total_lines}")
    print(f"有效记录数: {valid_lines}")
    print(f"格式正确率: {(valid_lines/max(total_lines, 1))*100:.2f}%")
    
    if errors:
        print(f"发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"  ❌ {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
        return False
    else:
        print("✅ 所有记录格式正确！")
        return True

def preview_data(jsonl_file: str, num_lines: int = 3):
    """预览数据"""
    print(f"\n📋 前{num_lines}条数据预览:")
    print("="*50)
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > num_lines:
                    break
                
                data = json.loads(line.strip())
                print(f"\n{i}. {json.dumps(data, ensure_ascii=False, indent=2)}")
    
    except Exception as e:
        print(f"预览数据时出错: {e}")

def check_data_statistics(jsonl_file: str):
    """检查数据统计信息"""
    print("\n📊 数据统计信息:")
    print("="*50)
    
    try:
        total_chars = 0
        human_chars = 0
        gpt_chars = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                for msg in data.get("conversations", []):
                    text = msg.get("value", "")
                    total_chars += len(text)
                    
                    if msg.get("from") == "human":
                        human_chars += len(text)
                    elif msg.get("from") == "gpt":
                        gpt_chars += len(text)
        
        print(f"总字符数: {total_chars:,}")
        print(f"human消息总字符数: {human_chars:,}")
        print(f"gpt消息总字符数: {gpt_chars:,}")
        print(f"平均每条记录字符数: {total_chars//(sum(1 for _ in open(jsonl_file)))}")
        
    except Exception as e:
        print(f"统计数据时出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        jsonl_file = sys.argv[1]
    else:
        jsonl_file = "datasets/job_skills_llamafactory_sft.jsonl"
    
    # 验证格式
    is_valid = validate_format(jsonl_file)
    
    if is_valid:
        preview_data(jsonl_file, 2)
        check_data_statistics(jsonl_file)
    else:
        sys.exit(1)