#!/usr/bin/env python3
"""
DCAS - 简化的职位能力要求提取器

目标：从职位名称和描述中提取能力要求词语，生成简单的数据集
格式：job_title, job_description, skill_requirements

使用DeepSeek作为教师模型提取关键能力词语
"""

import pandas as pd
import requests
import time
import os
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSkillExtractor:
    """简化的技能提取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def extract_skills(self, job_title: str, job_description: str) -> str:
        """
        提取职位能力要求
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            str: 能力要求词语，用逗号分隔
        """
        prompt = f"""
根据以下职位信息，提取出该职位需要的核心能力要求。请只返回能力要求词语，用逗号分隔。

职位名称: {job_title}
职位描述: {job_description}

请提取以下类型的能力要求：
1. 技术技能（如：Python, Java, 数据分析, 项目管理等）
2. 软技能（如：沟通能力, 团队合作, 领导力等）
3. 专业技能（如：财务分析, 法律知识, 医疗技能等）
4. 工具技能（如：Excel, Photoshop, AutoCAD等）

只返回能力要求词语，用逗号分隔，不要其他文字。
例如：Python编程, 数据分析, 沟通能力, 项目管理, Excel
"""
        
        try:
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是专业的职位分析师，专门提取职位的能力要求关键词。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 300
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
    
    def process_dataset(self, input_file: str, output_file: str, max_records: int = 20):
        """
        处理数据集
        
        Args:
            input_file: 输入CSV文件
            output_file: 输出CSV文件
            max_records: 最大处理记录数
        """
        logger.info("开始处理数据集...")
        
        # 加载数据
        df = pd.read_csv(input_file)
        logger.info(f"加载了 {len(df)} 条记录")
        
        # 过滤和限制
        df = df.dropna(subset=['title', 'description'])
        df = df[df['description'].str.len() > 100]
        df = df.head(max_records)
        
        logger.info(f"过滤后处理 {len(df)} 条记录")
        
        # 准备结果数据
        results = []
        
        for index, row in df.iterrows():
            try:
                job_title = row['title']
                job_description = row['description']
                
                logger.info(f"处理 {index + 1}/{len(df)}: {job_title}")
                
                # 提取技能要求
                skills = self.extract_skills(job_title, job_description)
                
                if skills:
                    results.append({
                        'job_title': job_title,
                        'job_description': job_description[:500],  # 截断描述
                        'skill_requirements': skills
                    })
                    logger.info(f"成功提取技能: {skills[:50]}...")
                
                # 延迟避免API限制
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"处理失败: {e}")
                continue
        
        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"处理完成！成功处理 {len(results)} 条记录")
        logger.info(f"输出文件: {output_file}")

def main():
    """主函数"""
    # 获取API密钥
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        logger.error("请设置DEEPSEEK_API_KEY环境变量")
        return
    
    # 文件路径
    input_file = "datasets/Job Descptions/postings.csv"
    output_file = f"datasets/job_skills_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # 创建处理器并处理数据
    extractor = SimpleSkillExtractor(api_key)
    extractor.process_dataset(input_file, output_file, max_records=200000000)
    
    logger.info("技能提取完成！")

if __name__ == "__main__":
    main()