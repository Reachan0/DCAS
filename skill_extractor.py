#!/usr/bin/env python3
"""
DCAS - 简化的职位能力要求提取器

目标：从职位名称和描述中提取能力要求词语，生成简单的数据集
格式：job_title, job_description, skill_requirements

使用Doubao-Seed-1.6-flash作为教师模型提取关键能力词语
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
from openai import OpenAI

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSkillExtractor:
    """简化的技能提取器（支持多线程，使用Doubao API）"""
    
    def __init__(self, api_key: str, max_workers: int = 5):
        self.api_key = api_key
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key
        )
        self.progress_file = "datasets/progress.json"
        self.temp_output_file = "datasets/temp_results.csv"
        self.max_workers = max_workers
        self.results_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.processed_count = 0
    
    def save_progress(self, current_index: int, total_records: int, results: list):
        """
        保存进度信息（线程安全）
        
        Args:
            current_index: 当前处理的索引
            total_records: 总记录数
            results: 当前结果列表
        """
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
                
                # 保存临时结果
                if results:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(self.temp_output_file, index=False, encoding='utf-8')
                    
            except Exception as e:
                logger.error(f"保存进度失败: {e}")
    
    def load_progress(self):
        """
        加载进度信息
        
        Returns:
            tuple: (起始索引, 已有结果列表)
        """
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                
                start_index = progress_data.get('current_index', 0)
                
                # 加载已有结果
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
        """
        获取已处理的职位ID集合
        
        Returns:
            set: 已处理的职位ID集合
        """
        try:
            if os.path.exists(self.temp_output_file):
                temp_df = pd.read_csv(self.temp_output_file)
                # 使用职位标题作为唯一标识
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

    def extract_skills(self, job_title: str, job_description: str) -> str:
        """
        提取职位能力要求（使用Doubao API，带随机延迟避免API限制）
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            str: 能力要求词语，用逗号分隔
        """
        # 截断过长的描述，减少token消耗
        description = job_description[:300] if len(job_description) > 300 else job_description
        
        prompt = f"""职位: {job_title}
描述: {description}

提取核心技能要求，用逗号分隔，只返回技能词语："""
        
        # 随机延迟避免并发API限制
        time.sleep(random.uniform(0.1, 0.5))
        
        try:
            response = self.client.chat.completions.create(
                model="doubao-seed-1-6-flash-250615",
                messages=[
                    {"role": "system", "content": "提取职位技能要求，只返回技能词语，用逗号分隔"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            skills = response.choices[0].message.content.strip()
            
            # 清理结果，确保只保留技能词语
            skills = skills.replace('、', ',').replace('，', ',')
            return skills
            
        except Exception as e:
            logger.error(f"提取技能失败: {e}")
            return ""
    
    def process_single_job(self, job_data):
        """
        处理单个职位（用于多线程）
        
        Args:
            job_data: 包含(index, job_title, job_description)的元组
            
        Returns:
            dict: 处理结果或None
        """
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
                    if self.processed_count % 10 == 0:  # 每10个记录输出一次进度
                        logger.info(f"线程处理进度: {self.processed_count} 条记录, 当前: {job_title}")
                
                return result
            
        except Exception as e:
            logger.error(f"处理职位失败 {job_title}: {e}")
        
        return None

    def extract_skills_old(self, job_title: str, job_description: str) -> str:
        """
        提取职位能力要求
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            str: 能力要求词语，用逗号分隔
        """
        # 截断过长的描述，减少token消耗
        description = job_description[:300] if len(job_description) > 300 else job_description
        
        prompt = f"""职位: {job_title}
描述: {description}

提取核心技能要求，用逗号分隔，只返回技能词语："""
        
        try:
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "提取职位技能要求，只返回技能词语，用逗号分隔"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 150  # 大幅减少max_tokens
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            skills = result['choices'][0]['message']['content'].strip()
            
            # 清理结果，确保只保留技能词语
            skills = skills.replace('、', ',').replace('，', ',')
            return skills
            
        except Exception as e:
            logger.error(f"提取技能失败: {e}")
            return ""
    
    def process_dataset(self, input_file: str, output_file: str, max_records: int = None, resume: bool = True, save_interval: int = 100):
        """
        处理数据集（多线程版本）
        
        Args:
            input_file: 输入CSV文件
            output_file: 输出CSV文件
            max_records: 最大处理记录数
            resume: 是否恢复之前的进度
            save_interval: 保存间隔（处理多少条记录后保存一次）
        """
        logger.info(f"开始多线程处理数据集（{self.max_workers} 个线程）...")
        
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
            
            # 跳过已处理的职位
            if job_title in processed_jobs:
                continue
                
            tasks.append((index, job_title, job_description))
        
        logger.info(f"准备处理 {len(tasks)} 个新任务")
        
        # 多线程处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.process_single_job, task): task 
                for task in tasks
            }
            
            completed_count = len(results)
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    if result:
                        with self.results_lock:
                            results.append(result)
                            completed_count = len(results)
                        
                        # 定期保存进度
                        if completed_count % save_interval == 0:
                            # 找到最大的已处理索引
                            max_index = max([t[0] for t in tasks[:completed_count]], default=start_index)
                            self.save_progress(max_index + 1, len(df), results)
                            logger.info(f"已保存进度，完成 {completed_count} 条记录")
                        
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"任务处理异常 {task[1]}: {e}")
        
        # 保存最终结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"多线程处理完成！成功处理 {len(results)} 条记录")
        logger.info(f"输出文件: {output_file}")
        
        # 清理临时文件
        self.clean_progress()
        
        return len(results)

    def process_dataset_old(self, input_file: str, output_file: str, max_records: int = None, resume: bool = True, save_interval: int = 100):
        """
        处理数据集
        
        Args:
            input_file: 输入CSV文件
            output_file: 输出CSV文件
            max_records: 最大处理记录数
            resume: 是否恢复之前的进度
            save_interval: 保存间隔（处理多少条记录后保存一次）
        """
        logger.info("开始处理数据集...")
        
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
        
        # 处理数据
        for index, row in df.iterrows():
            try:
                # 跳过已处理的职位
                if index < start_index:
                    continue
                
                job_title = row['title']
                job_description = row['description']
                
                # 增量更新：跳过已处理的职位
                if job_title in processed_jobs:
                    logger.info(f"跳过已处理职位: {job_title}")
                    continue
                
                logger.info(f"处理 {index + 1}/{len(df)}: {job_title}")
                
                # 提取技能要求
                skills = self.extract_skills(job_title, job_description)
                
                if skills:
                    results.append({
                        'job_title': job_title,
                        'job_description': job_description[:200],  # 进一步缩短描述
                        'skill_requirements': skills
                    })
                    processed_jobs.add(job_title)
                    logger.info(f"成功提取技能: {skills[:50]}...")
                
                # 定期保存进度
                if len(results) % save_interval == 0:
                    self.save_progress(index + 1, len(df), results)
                    logger.info(f"已保存进度，当前处理了 {len(results)} 条记录")
                
                # 延迟避免API限制 - 减少延迟时间
                time.sleep(0.5)  # 从1秒减少到0.5秒
                
            except Exception as e:
                logger.error(f"处理失败: {e}")
                # 即使出错也要保存进度
                self.save_progress(index + 1, len(df), results)
                continue
        
        # 保存最终结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"处理完成！成功处理 {len(results)} 条记录")
        logger.info(f"输出文件: {output_file}")
        
        # 清理临时文件
        self.clean_progress()
        
        return len(results)

def main():
    """主函数"""
    # 获取API密钥
    api_key = os.getenv('ARK_API_KEY')
    if not api_key:
        logger.error("请设置ARK_API_KEY环境变量")
        return
    
    # 文件路径
    input_file = "datasets/Job Descptions/postings.csv"
    output_file = f"datasets/job_skills_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # 多线程配置
    max_workers = 10  # Doubao API可以支持更多并发
    
    # 创建处理器
    extractor = SimpleSkillExtractor(api_key, max_workers=max_workers)
    
    # 处理数据集参数
    max_records = 200000000  # 处理所有记录
    resume = True  # 启用恢复功能
    save_interval = 50  # 每50条记录保存一次
    
    logger.info(f"使用Doubao-Seed-1.6-flash API，{max_workers} 个线程并发处理")
    
    try:
        # 处理数据集
        processed_count = extractor.process_dataset(
            input_file, 
            output_file, 
            max_records=max_records,
            resume=resume,
            save_interval=save_interval
        )
        logger.info(f"多线程技能提取完成！总共处理了 {processed_count} 条记录")
        
    except KeyboardInterrupt:
        logger.info("用户中断处理，进度已保存")
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        logger.info("进度已保存，可以重新运行脚本继续处理")

if __name__ == "__main__":
    main()