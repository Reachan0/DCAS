#!/usr/bin/env python3
"""
DCAS - 职位能力要求提取器（MLX版本）

专门使用mlx_lm库加载Qwen/Qwen3-30B-A3B-MLX-4bit模型
根据官方文档正确加载和使用MLX量化模型
"""

import pandas as pd
import time
import os
import logging
from datetime import datetime
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalMLXSkillExtractor:
    """使用MLX模型的本地技能提取器"""
    
    def __init__(self, max_workers: int = 2):
        """
        初始化本地MLX模型提取器
        
        Args:
            max_workers: 最大并发数（MLX模型建议较少）
        """
        self.progress_file = "datasets/progress.json"
        self.temp_output_file = "datasets/temp_results.csv"
        self.max_workers = max_workers
        self.results_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.processed_count = 0
        self.model = None
        self.tokenizer = None
        self.generate_func = None
        self.model_name = None
        
        # 初始化MLX模型
        self._load_mlx_model()
    
    def _load_mlx_model(self):
        """加载MLX模型"""
        try:
            logger.info("正在加载Qwen3-30B-A3B-MLX-4bit模型...")
            
            # 设置MLX环境变量
            os.environ['MLXLM_USE_MODELSCOPE'] = 'True'
            
            # 使用mlx_lm加载模型
            try:
                from mlx_lm import load, generate
                
                model_name = "Qwen/Qwen3-30B-A3B-MLX-4bit"
                logger.info(f"正在使用mlx_lm加载模型: {model_name}")
                
                # 加载模型和tokenizer
                self.model, self.tokenizer = load(model_name)
                self.model_name = model_name
                self.generate_func = generate  # 保存generate函数
                
                logger.info("MLX模型加载成功！")
                return
                
            except ImportError:
                logger.error("mlx_lm未安装！请运行: pip install --upgrade mlx_lm")
                raise Exception("需要安装mlx_lm库")
            except Exception as e:
                logger.error(f"MLX模型加载失败: {e}")
                raise e
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e
    
    def extract_skills_from_job(self, job_title: str, job_description: str) -> str:
        """
        使用MLX模型从职位信息中提取技能要求
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            str: 提取的技能要求，逗号分隔
        """
        try:
            # 构建prompt
            prompt = f"""请分析以下职位信息，提取出该职位需要的核心技能要求。请只返回技能关键词，用逗号分隔，不要包含其他内容。

职位名称: {job_title}
职位描述: {job_description}

核心技能要求:"""

            # 使用聊天模板
            if self.tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False  # 禁用思考模式，提高效率
                )
            else:
                formatted_prompt = prompt

            # 生成响应
            response = self.generate_func(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                verbose=False,
                max_tokens=512,  # 技能提取不需要太长
                temperature=0.3,  # 降低温度，提高一致性
                top_p=0.8,
                top_k=20
            )
            
            # 清理响应
            skills = self._clean_skills_response(response)
            return skills
            
        except Exception as e:
            logger.error(f"技能提取失败: {e}")
            return "提取失败"
    
    def _clean_skills_response(self, response: str) -> str:
        """清理模型响应，提取纯净的技能列表"""
        try:
            # 移除可能的思考标签
            response = response.replace("<think>", "").replace("</think>", "")
            
            # 按行分割，找到最相关的行
            lines = response.strip().split('\n')
            skills_line = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 跳过明显的解释性文本
                if any(word in line.lower() for word in ['分析', '总结', '职位', '要求', '包括', '需要']):
                    continue
                
                # 如果包含逗号分隔的技能，很可能是我们要的
                if ',' in line and len(line.split(',')) > 1:
                    skills_line = line
                    break
                elif not skills_line:  # 如果还没找到，记录第一个可能的行
                    skills_line = line
            
            if not skills_line:
                skills_line = lines[0] if lines else response
            
            # 清理技能列表
            skills = []
            for skill in skills_line.split(','):
                skill = skill.strip()
                # 移除序号
                skill = skill.lstrip('0123456789.- ')
                # 移除引号
                skill = skill.strip('"\'')
                
                if skill and len(skill) > 1:
                    skills.append(skill)
            
            return ', '.join(skills[:10])  # 限制最多10个技能
            
        except Exception as e:
            logger.error(f"清理响应失败: {e}")
            return response[:100]  # 返回前100个字符作为备选
    
    def process_batch(self, jobs_batch):
        """处理一批职位数据"""
        results = []
        
        for job in jobs_batch:
            try:
                job_id = job.get('job_id', '')
                company_name = job.get('company_name', '')
                title = job.get('title', '')
                description = job.get('description', '')
                
                # 清理数据
                title = str(title).strip() if title else ''
                description = str(description).strip() if description else ''
                
                if not title or not description:
                    logger.warning(f"跳过无效数据: job_id={job_id}")
                    continue
                
                # 提取技能
                skills = self.extract_skills_from_job(title, description)
                
                result = {
                    'job_title': title,
                    'job_description': description,
                    'skill_requirements': skills
                }
                
                results.append(result)
                
                # 增加处理计数
                with self.results_lock:
                    self.processed_count += 1
                
                logger.info(f"已处理 {self.processed_count} 条记录 - {title}: {skills}")
                
                # 短暂休息，避免过载
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"处理记录失败: {e}")
                continue
        
        return results
    
    def save_progress(self, processed_jobs, output_file):
        """保存进度"""
        try:
            with self.progress_lock:
                # 保存结果到CSV
                if processed_jobs:
                    df = pd.DataFrame(processed_jobs)
                    
                    # 如果文件存在，追加；否则创建
                    if os.path.exists(output_file):
                        df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
                    else:
                        df.to_csv(output_file, index=False, encoding='utf-8')
                
                # 保存进度信息
                progress = {
                    'processed_count': self.processed_count,
                    'last_update': datetime.now().isoformat(),
                    'model_name': self.model_name
                }
                
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def load_progress(self):
        """加载进度"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.processed_count = progress.get('processed_count', 0)
                    logger.info(f"已加载进度: {self.processed_count} 条记录已处理")
                    return progress
            return {}
        except Exception as e:
            logger.warning(f"加载进度失败: {e}")
            return {}
    
    def process_dataset(self, input_file: str, output_file: str = None, batch_size: int = 10):
        """
        处理整个数据集
        
        Args:
            input_file: 输入CSV文件路径
            output_file: 输出文件路径（可选）
            batch_size: 批处理大小
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"datasets/job_skills_mlx_{timestamp}.csv"
        
        logger.info(f"开始处理数据集: {input_file}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"使用模型: {self.model_name}")
        
        # 加载进度
        self.load_progress()
        
        try:
            # 读取数据
            df = pd.read_csv(input_file)
            logger.info(f"总共 {len(df)} 条记录")
            
            # 跳过已处理的记录
            if self.processed_count > 0:
                df = df.iloc[self.processed_count:]
                logger.info(f"跳过已处理的 {self.processed_count} 条记录，剩余 {len(df)} 条")
            
            # 分批处理
            all_results = []
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 条记录)")
                
                # 转换为字典列表
                jobs_batch = batch.to_dict('records')
                
                # 处理批次
                batch_results = self.process_batch(jobs_batch)
                all_results.extend(batch_results)
                
                # 每处理一个批次就保存一次
                self.save_progress(batch_results, output_file)
                
                logger.info(f"批次 {batch_num} 完成，累计处理 {self.processed_count} 条记录")
            
            logger.info(f"处理完成！总共处理 {len(all_results)} 条有效记录")
            logger.info(f"结果保存到: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"处理数据集失败: {e}")
            raise e

def main():
    """主函数"""
    print("=== DCAS 本地MLX技能提取器 ===")
    
    # 检查输入文件
    input_file = "datasets/Job Descptions/postings.csv"  # 修正拼写错误
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    try:
        # 询问参数
        print(f"\n输入文件: {input_file}")
        
        # 设置参数
        max_workers = 1  # MLX模型单线程处理更稳定
        batch_size = 5   # 小批次，避免内存问题
        
        print(f"并发数: {max_workers}")
        print(f"批处理大小: {batch_size}")
        
        # 创建提取器
        extractor = LocalMLXSkillExtractor(max_workers=max_workers)
        
        # 处理数据集
        output_file = extractor.process_dataset(
            input_file=input_file,
            batch_size=batch_size
        )
        
        print(f"\n✅ 处理完成！")
        print(f"📁 结果文件: {output_file}")
        print(f"🔥 使用模型: {extractor.model_name}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断处理")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise e

if __name__ == "__main__":
    main()