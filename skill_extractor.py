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
from volcenginesdkarkruntime import Ark

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSkillExtractor:
    """简化的技能提取器（支持多线程，使用Volcengine Ark）"""
    
    def __init__(self, api_key: str, max_workers: int = 5):
        self.api_key = api_key
        self.client = Ark(
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
        提取职位能力要求（使用Volcengine Ark，带随机延迟避免API限制）
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            str: 能力要求词语，用逗号分隔
        """
        # 截断过长的描述，减少token消耗
        description = job_description[:300] if len(job_description) > 300 else job_description
        
        # 使用英文prompt，确保返回英文技能
        prompt = f"""Job Title: {job_title}
Job Description: {description}

Extract the core skill requirements for this position. Return ONLY skill keywords in English, separated by commas. Do not include explanations, job descriptions, or the word "none". Focus on technical skills, soft skills, and qualifications.

Examples of good output: "Python, Data Analysis, Machine Learning, Communication, Project Management"

Skills:"""
        
        # 随机延迟避免并发API限制
        time.sleep(random.uniform(0.1, 0.5))
        
        try:
            response = self.client.chat.completions.create(
                model="doubao-seed-1-6-flash-250615",
                messages=[
                    {"role": "system", "content": "You are a professional recruiter. Extract job skill requirements and return ONLY English skill keywords separated by commas. Never return 'none', 'no skills', or any explanatory text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            skills = response.choices[0].message.content.strip()
            
            # 清理和验证结果
            skills = self._clean_and_validate_skills(skills)
            return skills
            
        except Exception as e:
            logger.error(f"提取技能失败: {e}")
            return ""
    
    def _clean_and_validate_skills(self, skills: str) -> str:
        """
        清理和验证技能文本，确保输出质量
        
        Args:
            skills: 原始技能文本
            
        Returns:
            str: 清洗后的技能文本
        """
        if not skills:
            return ""
        
        # 转换为小写进行检查
        skills_lower = skills.lower().strip()
        
        # 拒绝无效回答
        invalid_responses = [
            "无", "none", "no skills", "n/a", "not specified", "not mentioned",
            "not applicable", "no specific skills", "no requirements", 
            "skills:", "技能:", "requirements:", "要求:"
        ]
        
        for invalid in invalid_responses:
            if skills_lower == invalid or skills_lower.startswith(invalid):
                logger.warning(f"检测到无效技能回答: {skills}")
                return ""
        
        # 清理标点符号和格式
        skills = skills.replace('、', ',').replace('，', ',').replace(';', ',')
        skills = skills.replace('\n', ',').replace('\r', ',')
        
        # 分割技能并清理
        skill_list = []
        for skill in skills.split(','):
            skill = skill.strip()
            
            # 移除编号和特殊字符
            skill = skill.lstrip('0123456789.- ')
            skill = skill.strip('"\'')
            
            # 跳过过短或无效的技能
            if len(skill) < 2:
                continue
                
            # 跳过明显的非技能文本
            if any(word in skill.lower() for word in [
                'job description', 'position', 'candidate', 'experience',
                'position requires', 'we are looking', 'ideal candidate'
            ]):
                continue
            
            # 标准化技能名称（首字母大写）
            if skill.islower():
                skill = skill.title()
            
            skill_list.append(skill)
        
        # 去重并限制数量
        unique_skills = list(dict.fromkeys(skill_list))  # 保持顺序的去重
        final_skills = unique_skills[:10]  # 最多保留10个技能
        
        result = ', '.join(final_skills)
        
        # 最终验证：确保不为空且有实际内容
        if not result or len(result) < 5:
            logger.warning(f"技能提取结果过短或无效: {result}")
            return ""
        
        return result
    
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
            
            # 只有当技能提取成功且有效时才保存
            if skills and len(skills.strip()) > 5:
                result = {
                    'job_title': job_title,
                    'job_description': job_description[:200],
                    'skill_requirements': skills
                }
                
                with self.results_lock:
                    self.processed_count += 1
                    if self.processed_count % 10 == 0:  # 每10个记录输出一次进度
                        logger.info(f"线程处理进度: {self.processed_count} 条记录, 当前: {job_title} -> {skills[:50]}")
                
                return result
            else:
                logger.warning(f"跳过无效技能提取结果: {job_title} -> '{skills}'")
            
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
    
    # 多线程配置（降低并发数，提高稳定性）
    max_workers = 5  # 降低并发数，避免API限制
    
    # 创建处理器
    extractor = SimpleSkillExtractor(api_key, max_workers=max_workers)
    
    # 处理数据集参数
    max_records = 200000000  # 处理所有记录
    resume = True  # 启用恢复功能
    save_interval = 20  # 每20条记录保存一次（更频繁保存）
    
    logger.info(f"使用Doubao-Seed-1.6-flash API，{max_workers} 个线程并发处理")
    logger.info("新增功能：自动过滤'无'和中英文混用问题")
    
    try:
        # 处理数据集
        processed_count = extractor.process_dataset(
            input_file, 
            output_file, 
            max_records=max_records,
            resume=resume,
            save_interval=save_interval
        )
        logger.info(f"技能提取完成！总共处理了 {processed_count} 条高质量记录")
        
        # 输出数据质量统计
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            logger.info(f"输出文件统计：")
            logger.info(f"- 总记录数: {len(df)}")
            logger.info(f"- 平均技能数: {df['skill_requirements'].str.count(',').mean() + 1:.1f}")
            logger.info(f"- 输出文件: {output_file}")
        
    except KeyboardInterrupt:
        logger.info("用户中断处理，进度已保存")
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        logger.info("进度已保存，可以重新运行脚本继续处理")

if __name__ == "__main__":
    main()