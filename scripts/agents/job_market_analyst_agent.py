#!/usr/bin/env python3
"""
就业市场分析智能体 - 推理时思维链(CoT)实现
"""

import re
import logging
from typing import Optional, Dict, Any
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobMarketAnalystAgent:
    """就业市场分析智能体 - 使用推理时CoT"""
    
    def __init__(self, model_client=None):
        """
        初始化分析智能体
        
        Args:
            model_client: 模型客户端，负责调用微调后的模型
        """
        self.model_client = model_client
        
        # CoT Prompt模板
        self.cot_prompt_template = """【背景】
你是一个经过微调的、专业的就业市场分析AI。你擅长从职位描述中精准地提取核心能力要求。

【任务】
现在，你需要分析一份新的职位信息。请严格遵循以下的【思考框架】，一步步地进行分析，并在最后给出结论。

【思考框架】
1.  **核心职责分析**: 首先，仔细阅读【输入信息】，用1-2句话总结这个职位的核心工作职责是什么。
2.  **硬技能与工具推断**: 基于核心职责，推断出完成这些工作所必需的硬技能（如编程语言、设计软件、专业知识领域等）和软件工具。
3.  **软技能与综合能力推断**: 分析描述中隐含的对候选人综合能力的要求（如沟通能力、团队协作、解决问题能力等）。
4.  **最终能力要求整合**: 综合以上所有分析，将所有识别出的能力要求，整合成一个最终的、由英文逗号和单个空格分隔的字符串列表。

---
【输入信息】:
职位名称: {job_title}
职位描述: {job_description}
---

【你的输出】
请严格按照以下格式填充你的分析结果：

**核心职责分析:**
[请在此处填充你的分析...]

**硬技能与工具推断:**
[请在此处填充你的分析...]

**软技能与综合能力推断:**
[请在此处填充你的分析...]

**最终能力要求列表:**
[请在此处填充最终的、逗号分隔的字符串...]"""
    
    def _build_cot_prompt(self, job_title: str, job_description: str) -> str:
        """
        构建CoT Prompt
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            完整的CoT Prompt字符串
        """
        return self.cot_prompt_template.format(
            job_title=job_title,
            job_description=job_description
        )
    
    def _parse_final_skills(self, cot_response: str) -> str:
        """
        从CoT响应中解析最终技能列表
        
        Args:
            cot_response: 模型返回的完整CoT响应文本
            
        Returns:
            最终的技能字符串（逗号分隔）
        """
        try:
            # 查找"最终能力要求列表:"后的内容
            pattern = r'\*\*最终能力要求列表:\*\*\s*([\s\S]*)'
            match = re.search(pattern, cot_response, re.IGNORECASE)
            
            if match:
                # 提取匹配的内容并清理
                skills_text = match.group(1).strip()
                
                # 移除可能的空行和多余空格
                skills_text = re.sub(r'\n+', ' ', skills_text)
                skills_text = re.sub(r'\s+', ' ', skills_text).strip()
                
                # 确保使用英文逗号分隔
                skills_text = skills_text.replace('，', ',')
                
                return skills_text
            else:
                # 如果没有找到标记，返回空字符串
                logger.warning("未找到'最终能力要求列表'标记")
                return ""
                
        except Exception as e:
            logger.error(f"解析CoT响应时出错: {e}")
            return ""
    
    def analyze(self, job_title: str, job_description: str) -> Dict[str, Any]:
        """
        使用CoT方法分析职位技能要求
        
        Args:
            job_title: 职位名称
            job_description: 职位描述
            
        Returns:
            包含完整分析结果的字典
        """
        if not self.model_client:
            raise ValueError("model_client未设置")
        
        # 构建CoT Prompt
        cot_prompt = self._build_cot_prompt(job_title, job_description)
        logger.info(f"构建CoT Prompt完成，长度: {len(cot_prompt)}")
        
        # 调用模型获取CoT响应
        try:
            cot_response = self.model_client.generate(cot_prompt)
            logger.info("模型调用成功")
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            raise
        
        # 解析最终技能列表
        final_skills = self._parse_final_skills(cot_response)
        
        # 返回完整结果
        return {
            "job_title": job_title,
            "job_description": job_description,
            "cot_prompt": cot_prompt,
            "cot_response": cot_response,
            "final_skills": final_skills,
            "success": bool(final_skills)
        }
    
    def set_model_client(self, model_client):
        """设置模型客户端"""
        self.model_client = model_client


class MockModelClient:
    """模拟模型客户端，用于测试"""
    
    def generate(self, prompt: str) -> str:
        """模拟模型生成响应"""
        # 从prompt中提取职位信息
        job_title_match = re.search(r'职位名称: (.+?)\n', prompt)
        job_desc_match = re.search(r'职位描述: (.+?)(?=---|$)', prompt, re.DOTALL)
        
        job_title = job_title_match.group(1).strip() if job_title_match else "未知职位"
        job_desc = job_desc_match.group(1).strip() if job_desc_match else "无描述"
        
        # 生成模拟的CoT响应
        return f"""**核心职责分析:**
该职位主要负责{job_title}相关的核心工作，包括日常运营、项目管理和团队协作。

**硬技能与工具推断:**
基于职位描述，需要掌握{job_title}相关的专业技能，熟悉行业标准工具，具备数据分析能力。

**软技能与综合能力推断:**
要求具备良好的沟通能力、团队协作精神、问题解决能力和项目管理经验。

**最终能力要求列表:**
{job_title.lower().replace(' ', '_')}, technical_skills, data_analysis, communication, teamwork, problem_solving"""


# 测试代码
if __name__ == "__main__":
    # 创建测试实例
    agent = JobMarketAnalystAgent()
    mock_client = MockModelClient()
    agent.set_model_client(mock_client)
    
    # 测试分析
    result = agent.analyze(
        job_title="Software Engineer",
        job_description="We are looking for a skilled software engineer to join our team. The ideal candidate will have experience in Python, JavaScript, and cloud technologies. You will be responsible for developing web applications, collaborating with cross-functional teams, and solving complex technical problems."
    )
    
    print("测试结果:")
    print(f"最终技能: {result['final_skills']}")
    print(f"成功: {result['success']}")