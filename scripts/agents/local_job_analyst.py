#!/usr/bin/env python3
"""
本地部署的就业市场分析智能体
结合JobMarketAnalystAgent与本地Qwen3模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from job_market_analyst_agent import JobMarketAnalystAgent
from typing import Optional, Dict, Any
import logging
import re
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalModelClient:
    """本地模型客户端，适配JobMarketAnalystAgent"""
    
    def __init__(self, model_path: str = None):
        """
        初始化本地模型客户端
        
        Args:
            model_path: 本地模型路径，如果为None则自动选择
        """
        self.model = None
        self.tokenizer = None
        self.model_path = model_path or self._get_default_model()
        self._load_model()
    
    def _get_default_model(self) -> str:
        """获取默认模型路径"""
        # 优先级：MLX > Transformers > 备选
        return None  # 让自动选择生效
    
    def _load_model(self):
        """加载本地模型"""
        try:
            logger.info("正在加载本地模型...")
            
            # 尝试多种加载方式
            
            # 方法1：MLX (Mac专用)
            try:
                import mlx.core as mx
                from mlx_lm import load, generate
                
                # 尝试多个MLX模型
                mlx_models = [
                    "mlx-community/Qwen2.5-7B-Instruct-4bit",
                    "mlx-community/Qwen2.5-3B-Instruct-4bit",
                    "mlx-community/Qwen-14B-Chat-4bit"
                ]
                
                for model_name in mlx_models:
                    try:
                        self.model, self.tokenizer = load(model_name)
                        self.use_mlx = True
                        self.model_name = model_name
                        logger.info(f"MLX模型加载成功: {model_name}")
                        return
                    except Exception as e:
                        logger.warning(f"MLX模型 {model_name} 加载失败: {e}")
                        continue
                        
            except ImportError:
                logger.info("MLX不可用，尝试Transformers...")
            
            # 方法2：Transformers
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                # 尝试多个模型
                transformer_models = [
                    "Qwen/Qwen2.5-7B-Instruct",
                    "Qwen/Qwen2.5-3B-Instruct", 
                    "Qwen/Qwen-14B-Chat",
                    "Qwen/Qwen-7B-Chat"
                ]
                
                for model_name in transformer_models:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            trust_remote_code=True
                        )
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                        
                        self.use_mlx = False
                        self.model_name = model_name
                        logger.info(f"Transformers模型加载成功: {model_name}")
                        return
                        
                    except Exception as e:
                        logger.warning(f"Transformers模型 {model_name} 加载失败: {e}")
                        continue
                        
            except ImportError:
                logger.error("Transformers不可用")
            
            raise RuntimeError("无法加载任何模型")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        生成响应（适配JobMarketAnalystAgent接口）
        
        Args:
            prompt: 输入提示
            
        Returns:
            模型生成的响应文本
        """
        try:
            if hasattr(self, 'use_mlx') and self.use_mlx:
                # 使用MLX生成
                from mlx_lm import generate
                
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=500,
                    temp=0.1
                )
                return response.strip()
            
            else:
                # 使用Transformers生成
                messages = [
                    {"role": "system", "content": "你是一个专业的职位分析师，请按照要求提供详细的思维链分析。"},
                    {"role": "user", "content": prompt}
                ]
                
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    text = prompt
                
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                
                import torch
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                return response.strip()
                
        except Exception as e:
            logger.error(f"模型生成失败: {e}")
            return "**最终能力要求列表:**\nanalysis_failed, communication, problem_solving"


class LocalJobAnalyst:
    """本地部署的就业市场分析系统"""
    
    def __init__(self, model_path: str = None):
        """
        初始化本地分析系统
        
        Args:
            model_path: 本地模型路径
        """
        self.model_client = LocalModelClient(model_path)
        self.agent = JobMarketAnalystAgent(self.model_client)
    
    def analyze_job(self, job_title: str, job_description: str) -> Dict[str, any]:
        """
        分析单个职位
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            分析结果字典
        """
        logger.info(f"开始分析职位: {job_title}")
        
        try:
            result = self.agent.analyze(job_title, job_description)
            logger.info(f"分析完成: {result['final_skills']}")
            return result
            
        except Exception as e:
            logger.error(f"分析失败: {e}")
            return {
                'job_title': job_title,
                'job_description': job_description,
                'final_skills': 'error_occurred',
                'success': False,
                'error': str(e)
            }
    
    def analyze_batch(self, jobs: list) -> list:
        """
        批量分析职位
        
        Args:
            jobs: 职位列表，每个元素为(job_title, job_description)元组
            
        Returns:
            分析结果列表
        """
        results = []
        
        for i, (title, desc) in enumerate(jobs):
            logger.info(f"批量处理 {i+1}/{len(jobs)}: {title}")
            result = self.analyze_job(title, desc)
            results.append(result)
        
        return results


def main():
    """测试主函数"""
    print("=== 本地就业市场分析智能体测试 ===")
    
    # 创建本地分析器
    analyst = LocalJobAnalyst()
    
    # 测试数据
    test_jobs = [
        (
            "数据科学家", 
            "负责构建机器学习模型，分析大数据，与产品团队合作，需要Python和SQL经验，具备统计分析和数据可视化能力"
        ),
        (
            "前端开发工程师",
            "开发响应式Web应用，使用React/Vue框架，与UI/UX设计师协作，需要HTML/CSS/JavaScript经验"
        ),
        (
            "产品经理",
            "负责产品规划和需求分析，协调开发团队，制定产品路线图，需要市场调研和项目管理经验"
        )
    ]
    
    # 分析测试
    results = analyst.analyze_batch(test_jobs)
    
    # 输出结果
    print("\n=== 分析结果 ===")
    for result in results:
        print(f"\n职位: {result['job_title']}")
        print(f"技能要求: {result['final_skills']}")
        print(f"成功: {result['success']}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"local_analysis_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()