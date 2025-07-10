#!/usr/bin/env python3
"""
DCAS - 简化的职位能力要求提取器（本地模型版本）

目标：从职位名称和描述中提取能力要求词语，生成简单的数据集
格式：job_title, job_description, skill_requirements

使用ModelScope的Qwen3-30B-A3B-MLX-4bit本地模型提取关键能力词语
"""

import pandas as pd
import time
import os
import logging
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalSkillExtractor:
    """本地技能提取器（使用Qwen3-30B-A3B-MLX-4bit）"""
    
    def __init__(self, max_workers: int = 3):
        """
        初始化本地模型提取器
        
        Args:
            max_workers: 最大并发数（本地模型建议较少）
        """
        self.progress_file = "datasets/progress.json"
        self.temp_output_file = "datasets/temp_results.csv"
        self.max_workers = max_workers
        self.results_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.processed_count = 0
        self.model = None
        self.tokenizer = None
        self.model_name = None  # 初始化模型名称
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self):
        """加载本地模型"""
        try:
            logger.info("正在加载本地模型...")
            
            # 方法1：使用ModelScope - 尝试多个模型
            try:
                from modelscope import AutoModelForCausalLM, AutoTokenizer
                
                # 模型选择优先级：从小到大
                model_options = [
                    "Qwen/Qwen3-8B-Instruct",  # 8B版本，更稳定
                    "Qwen/Qwen3-14B-Instruct",  # 14B版本
                    "Qwen/Qwen3-30B-A3B-MLX-4bit"  # 原始30B量化版本
                ]
                
                for model_name in model_options:
                    try:
                        logger.info(f"尝试加载模型: {model_name}")
                        
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            trust_remote_code=True
                        )
                        
                        # 根据模型类型调整加载参数
                        if "MLX" in model_name:
                            # 量化模型的特殊处理
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                device_map="auto",
                                torch_dtype="auto",
                                ignore_mismatched_sizes=True,
                                low_cpu_mem_usage=True,
                                # 跳过量化配置验证
                                attn_implementation="eager"
                            )
                        else:
                            # 标准模型加载
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                device_map="auto",
                                torch_dtype="auto",
                                low_cpu_mem_usage=True
                            )
                        
                        logger.info(f"ModelScope模型加载成功: {model_name}")
                        self.model_name = model_name
                        return
                        
                    except Exception as e:
                        logger.warning(f"模型 {model_name} 加载失败: {e}")
                        continue
                
                raise Exception("所有ModelScope模型都加载失败")
                
            except ImportError:
                logger.warning("ModelScope未安装，尝试使用transformers...")
            
            # 方法2：使用transformers（备选）
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                model_name = "Qwen/Qwen2.5-7B-Instruct"  # 备选更小的模型
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                logger.info("Transformers模型加载成功")
                return
                
            except ImportError:
                logger.error("无法导入transformers库")
            
            # 方法3：使用MLX（Mac专用）
            try:
                import mlx.core as mx
                from mlx_lm import load, generate
                
                self.model, self.tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
                self.use_mlx = True
                
                logger.info("MLX模型加载成功")
                return
                
            except ImportError:
                logger.warning("MLX未安装或不支持")
            
            raise RuntimeError("无法加载任何模型，请安装相应依赖")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def extract_skills(self, job_title: str, job_description: str) -> str:
        """
        使用本地模型提取职位能力要求
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            str: 能力要求词语，用逗号分隔
        """
        # 截断过长的描述
        description = job_description[:300] if len(job_description) > 300 else job_description
        
        prompt = f"""你是一个专业的职位分析师。请根据以下职位信息，提取出该职位需要的核心技能要求。

职位名称: {job_title}
职位描述: {description}

请只返回技能要求关键词，用逗号分隔。例如：Python编程, 数据分析, 沟通能力, 项目管理

技能要求："""
        
        try:
            if hasattr(self, 'use_mlx') and self.use_mlx:
                # 使用MLX生成
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=100,
                    temp=0.1
                )
                skills = response.strip()
            else:
                # 使用transformers/modelscope生成
                messages = [
                    {"role": "system", "content": "你是专业的职位分析师，只返回技能要求关键词，用逗号分隔。"},
                    {"role": "user", "content": prompt}
                ]
                
                # 格式化输入
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    text = prompt
                
                # 编码
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # 生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                skills = response.strip()
            
            # 清理结果
            skills = skills.replace('、', ',').replace('，', ',')
            
            # 提取第一行（避免多余输出）
            if '\n' in skills:
                skills = skills.split('\n')[0]
            
            return skills
            
        except Exception as e:
            logger.error(f"提取技能失败: {e}")
            return ""
    
    def save_progress(self, current_index: int, total_records: int, results: list):
        """保存进度信息（线程安全）"""
        with self.progress_lock:
            progress_data = {
                "current_index": current_index,
                "total_records": total_records,
                "processed_count": len(results),
                "last_update": datetime.now().isoformat(),
                "temp_file": self.temp_output_file
            }
            
            try:
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)
                
                if results:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(self.temp_output_file, index=False, encoding='utf-8')
                    
            except Exception as e:
                logger.error(f"保存进度失败: {e}")
    
    def load_progress(self):
        """加载进度信息"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                
                start_index = progress_data.get('current_index', 0)
                
                results = []
                if os.path.exists(self.temp_output_file):
                    temp_df = pd.read_csv(self.temp_output_file)
                    results = temp_df.to_dict('records')
                
                logger.info(f"加载进度: 从索引 {start_index} 开始，已有 {len(results)} 条结果")
                return start_index, results
            else:
                logger.info("没有找到进度文件，从头开始")
                return 0, []
                
        except Exception as e:
            logger.error(f"加载进度失败: {e}")
            return 0, []
    
    def get_processed_job_ids(self):
        """获取已处理的职位ID集合"""
        try:
            if os.path.exists(self.temp_output_file):
                temp_df = pd.read_csv(self.temp_output_file)
                return set(temp_df['job_title'].tolist())
            return set()
        except Exception as e:
            logger.error(f"获取已处理职位ID失败: {e}")
            return set()
    
    def clean_progress(self):
        """清理进度文件"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            if os.path.exists(self.temp_output_file):
                os.remove(self.temp_output_file)
            logger.info("已清理进度文件")
        except Exception as e:
            logger.error(f"清理进度文件失败: {e}")
    
    def process_single_job(self, job_data):
        """处理单个职位（用于多线程）"""
        index, job_title, job_description = job_data
        
        try:
            # 提取技能要求
            skills = self.extract_skills(job_title, job_description)
            
            if skills:
                result = {
                    'job_title': job_title,
                    'job_description': job_description[:200],
                    'skill_requirements': skills
                }
                
                with self.results_lock:
                    self.processed_count += 1
                    if self.processed_count % 5 == 0:  # 本地模型处理较慢，减少日志频率
                        logger.info(f"本地模型处理进度: {self.processed_count} 条记录, 当前: {job_title}")
                
                return result
            
        except Exception as e:
            logger.error(f"处理职位失败 {job_title}: {e}")
        
        return None
    
    def process_dataset(self, input_file: str, output_file: str, max_records: int = None, resume: bool = True, save_interval: int = 20):
        """处理数据集（本地模型版本）"""
        logger.info(f"开始本地模型处理数据集（{self.max_workers} 个线程）...")
        
        # 加载数据
        df = pd.read_csv(input_file)
        logger.info(f"加载了 {len(df)} 条记录")
        
        # 过滤和限制
        df = df.dropna(subset=['title', 'description'])
        df = df[df['description'].str.len() > 100]
        if max_records:
            df = df.head(max_records)
        
        logger.info(f"过滤后处理 {len(df)} 条记录")
        
        # 加载进度
        start_index = 0
        results = []
        processed_jobs = set()
        
        if resume:
            start_index, results = self.load_progress()
            processed_jobs = self.get_processed_job_ids()
            logger.info(f"已处理 {len(processed_jobs)} 个职位")
        
        # 准备待处理的任务
        tasks = []
        for index, row in df.iterrows():
            if index < start_index:
                continue
                
            job_title = row['title']
            job_description = row['description']
            
            if job_title in processed_jobs:
                continue
                
            tasks.append((index, job_title, job_description))
        
        logger.info(f"准备处理 {len(tasks)} 个新任务")
        
        # 多线程处理（本地模型使用较少线程）
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.process_single_job, task): task 
                for task in tasks
            }
            
            completed_count = len(results)
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    if result:
                        with self.results_lock:
                            results.append(result)
                            completed_count = len(results)
                        
                        # 定期保存进度
                        if completed_count % save_interval == 0:
                            max_index = max([t[0] for t in tasks[:completed_count]], default=start_index)
                            self.save_progress(max_index + 1, len(df), results)
                            logger.info(f"已保存进度，完成 {completed_count} 条记录")
                        
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"任务处理异常 {task[1]}: {e}")
        
        # 保存最终结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"本地模型处理完成！成功处理 {len(results)} 条记录")
        logger.info(f"输出文件: {output_file}")
        
        # 清理临时文件
        self.clean_progress()
        
        return len(results)

def main():
    """主函数"""
    # 文件路径
    input_file = "datasets/Job Descptions/postings.csv"
    output_file = f"datasets/job_skills_dataset_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # 本地模型配置
    max_workers = 2  # 本地模型建议使用较少线程
    
    # 创建处理器
    extractor = LocalSkillExtractor(max_workers=max_workers)
    
    # 处理数据集参数
    max_records = 100  # 测试用，可以改为None处理全部
    resume = True
    save_interval = 10  # 本地模型处理较慢，更频繁保存
    
    logger.info(f"使用本地Qwen3模型，{max_workers} 个线程并发处理")
    
    try:
        processed_count = extractor.process_dataset(
            input_file, 
            output_file, 
            max_records=max_records,
            resume=resume,
            save_interval=save_interval
        )
        logger.info(f"本地模型技能提取完成！总共处理了 {processed_count} 条记录")
        
    except KeyboardInterrupt:
        logger.info("用户中断处理，进度已保存")
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        logger.info("进度已保存，可以重新运行脚本继续处理")

if __name__ == "__main__":
    main()