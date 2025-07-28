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
        
        # CoT Prompt模板 - 优化版
        self.cot_prompt_template = """【背景】
你是一个经过微调的专业就业市场分析师，擅长通过深度思维链分析从职位描述中精准提取核心技能要求。

【任务】
现在需要分析以下职位信息，请严格按照思维链逐步推理，最终生成针对该职位的精准技能清单。

【思维链框架】
1. **职位类型识别**:
   - 首先判断这是哪类技术职位（数据科学/前端/后端/产品/设计/运维等）
   - 识别职位的核心领域和技术栈方向

2. **技术技能深度分析**:
   - 识别所有提及的编程语言、框架、工具
   - 提取数据相关技术（数据库、分析工具、云平台等）
   - 识别开发相关技术（前端/后端/全栈技术栈）
   - 提取基础设施相关技能（CI/CD、云服务、容器化等）

3. **业务技能映射**:
   - 根据职位描述提取业务领域知识（金融、电商、医疗等）
   - 识别数据分析、产品设计、系统架构等业务能力

4. **软技能与协作要求**:
   - 提取团队协作、沟通、项目管理等软技能
   - 识别跨部门协作、用户研究、需求分析等综合能力

5. **技能优先级排序**:
   - 按重要性排序：核心技术技能 → 业务技能 → 软技能
   - 确保技能与职位描述高度匹配

【输出格式要求】
请完成完整的思维链分析，最后以以下格式输出技能列表：
**最终能力要求列表:** [在此列出技能，用英文逗号分隔]

---
【输入信息】:
职位名称: {job_title}
职位描述: {job_description}

【分析结果】:
**职位类型识别:**
[分析职位类型和领域]

**技术技能深度分析:**
[详细技术栈分析]

**业务技能映射:**
[业务领域能力]

**软技能与协作要求:**
[软技能清单]

**最终能力要求列表:** [用英文逗号分隔的精准技能清单]"""
    
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
        从CoT响应中解析最终技能列表 - 增强版解析
        
        Args:
            cot_response: 模型返回的完整思维链分析文本
            
        Returns:
            最终的技能字符串（逗号分隔）
        """
        try:
            response_text = cot_response.strip()
            
            # 从思维链中提取技能信息
            skills = []
            
            # 1. 从"最终能力要求列表:"中提取
            pattern = r'\*\*最终能力要求列表:\*\*\s*([\s\S]*?)(?:\n\s*\*\*|$)'
            match = re.search(pattern, response_text, re.IGNORECASE)
            
            if match:
                skills_text = match.group(1).strip()
                skills_text = re.sub(r'\n+', ' ', skills_text)
                skills_text = re.sub(r'\s+', ' ', skills_text).strip()
                skills_text = skills_text.replace('，', ',')
                
                if skills_text and len(skills_text) > 3:
                    # 清理格式并返回
                    skills_list = [s.strip() for s in skills_text.split(',') if s.strip()]
                    if skills_list:
                        return ', '.join(skills_list)
            
            # 2. 从思维链中提取具体技能
            # 技术技能提取
            tech_skills = re.findall(r'\b(Python|Java|JavaScript|SQL|MySQL|PostgreSQL|MongoDB|Redis|React|Vue|Angular|Node\.js|Spring|Django|Flask|TensorFlow|PyTorch|Keras|Pandas|NumPy|Scikit-learn|Docker|Kubernetes|AWS|Azure|GCP|Linux|Git|Jenkins|CI/CD|微服务|API|RESTful|GraphQL)\b', response_text, re.IGNORECASE)
            skills.extend(tech_skills)
            
            # 业务技能提取
            business_skills = re.findall(r'\b(数据分析|数据可视化|机器学习|深度学习|人工智能|统计学|数据挖掘|用户研究|产品管理|项目管理|需求分析|商业分析|系统架构|数据库设计|性能优化)\b', response_text, re.IGNORECASE)
            skills.extend(business_skills)
            
            # 软技能提取
            soft_skills = re.findall(r'\b(团队合作|沟通能力|问题解决|学习能力|责任心|创新思维|抗压能力|时间管理|跨部门协作|用户沟通)\b', response_text, re.IGNORECASE)
            skills.extend(soft_skills)
            
            # 3. 去重并返回
            if skills:
                return ', '.join(set([s.title() for s in skills]))
            
            # 4. 兜底方案 - 返回通用技能
            return "Python, SQL, 团队协作, 问题解决, 学习能力"
            
        except Exception as e:
            logger.error(f"解析CoT响应时出错: {e}")
            return "Python, SQL, 团队协作, 问题解决, 学习能力"

    def _get_position_skills(self, job_title: str) -> str:
        """根据职位类型返回默认技能"""
        job_type_mapping = {
            "数据科学家": "Python, SQL, 机器学习, 数据可视化, 统计学, 数据分析",
            "前端开发": "JavaScript, HTML, CSS, React, Vue, TypeScript, 前端框架",
            "后端开发": "Java, Spring, MySQL, Redis, Linux, Docker, 微服务架构",
            "产品经理": "需求分析, 产品管理, 用户研究, 项目管理, 商业分析",
            "UI设计师": "Figma, Sketch, Photoshop, 用户界面设计, 交互设计",
            "UX设计师": "用户研究, 原型设计, 交互设计, 用户体验, 可用性测试",
            "DevOps": "Docker, Kubernetes, Jenkins, Linux, AWS, CI/CD"
        }
        
        for job_type, skills in job_type_mapping.items():
            if job_type in job_title:
                return skills
        
        return "Python, SQL, 编程基础, 团队协作, 问题解决"
    
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
        
        logger.info(f"📋 开始分析职位: {job_title}")
        logger.info(f"📋 职位描述: {job_description[:100]}...")
        
        # 构建CoT Prompt
        cot_prompt = self._build_cot_prompt(job_title, job_description)
        logger.info(f"✅ 构建CoT Prompt完成，长度: {len(cot_prompt)}")
        logger.info(f"🔍 Prompt预览:\n{cot_prompt[:300]}...")
        
        # 调用模型获取CoT响应
        try:
            logger.info("🤖 开始调用模型进行推理...")
            import time
            start_time = time.time()
            
            cot_response = self.model_client.generate(cot_prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"✅ 模型推理完成，耗时: {inference_time:.2f}秒")
            logger.info(f"📝 模型响应长度: {len(cot_response)} 字符")
            logger.info(f"🧠 思维链响应:\n{'-'*50}\n{cot_response}\n{'-'*50}")
            
        except Exception as e:
            logger.error(f"❌ 模型调用失败: {e}")
            raise
        
        # 解析最终技能列表
        logger.info("🔧 开始解析技能列表...")
        final_skills = self._parse_final_skills(cot_response)
        logger.info(f"🎯 解析得到技能: {final_skills}")
        
        # 返回完整结果
        result = {
            "job_title": job_title,
            "job_description": job_description,
            "cot_prompt": cot_prompt,
            "cot_response": cot_response,
            "final_skills": final_skills,
            "success": bool(final_skills)
        }
        
        logger.info(f"✅ 分析完成，成功: {result['success']}")
        return result
    
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