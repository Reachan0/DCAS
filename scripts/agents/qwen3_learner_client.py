#!/usr/bin/env python3
"""
Qwen3客户端用于学习者画像智能体
实现高质量的学习风格推断
"""

import requests
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Qwen3LearnerProfileClient:
    """Qwen3专用的学习者画像分析客户端"""
    
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.session = requests.Session()
        
    def generate(self, prompt: str) -> str:
        """调用Qwen3 API生成学习风格分析"""
        try:
            response = self.session.post(
                f"{self.api_url}/analyze/quick", 
                json={
                    "job_title": "学习风格分析",
                    "job_description": prompt
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # 从技能分析结果中提取学习风格特征
                cot_response = result.get('cot_response', '')
                return self._extract_learning_style_from_cot(cot_response)
            else:
                logger.error(f"API调用失败: {response.status_code}")
                return self._fallback_learning_style()
                
        except Exception as e:
            logger.error(f"Qwen3 API调用异常: {e}")
            return self._fallback_learning_style()
    
    def _extract_learning_style_from_cot(self, cot_response: str) -> str:
        """从CoT响应中提取学习风格JSON"""
        # 基于CoT分析内容推断学习风格
        processing = self._infer_processing_style(cot_response)
        perception = self._infer_perception_style(cot_response)
        understanding = self._infer_understanding_style(cot_response)
        
        return json.dumps({
            "processing": processing,
            "perception": perception,
            "understanding": understanding
        })
    
    def _infer_processing_style(self, text: str) -> str:
        """从文本推断信息处理风格"""
        active_keywords = ["实践", "动手", "讨论", "试试", "实验", "项目", "操作"]
        reflective_keywords = ["思考", "独立", "分析", "理解", "消化", "反思", "琢磨"]
        
        active_score = sum(1 for keyword in active_keywords if keyword in text)
        reflective_score = sum(1 for keyword in reflective_keywords if keyword in text)
        
        return "active" if active_score >= reflective_score else "reflective"
    
    def _infer_perception_style(self, text: str) -> str:
        """从文本推断信息感知风格"""
        sensory_keywords = ["具体", "实际", "案例", "现实", "细节", "事实", "经验"]
        intuitive_keywords = ["抽象", "概念", "理论", "创新", "可能", "全局", "框架"]
        
        sensory_score = sum(1 for keyword in sensory_keywords if keyword in text)
        intuitive_score = sum(1 for keyword in intuitive_keywords if keyword in text)
        
        return "sensory" if sensory_score >= intuitive_score else "intuitive"
    
    def _infer_understanding_style(self, text: str) -> str:
        """从文本推断信息理解风格"""
        sequential_keywords = ["步骤", "顺序", "逐步", "线性", "有序", "一步步", "循序"]
        global_keywords = ["整体", "全局", "框架", "跳跃", "直觉", "大局", "宏观"]
        
        sequential_score = sum(1 for keyword in sequential_keywords if keyword in text)
        global_score = sum(1 for keyword in global_keywords if keyword in text)
        
        return "sequential" if sequential_score >= global_score else "global"
    
    def _fallback_learning_style(self) -> str:
        """当API不可用时的默认学习风格"""
        return json.dumps({
            "processing": "active", 
            "perception": "sensory", 
            "understanding": "sequential"
        })

class EnhancedProfileGenerator:
    """增强版画像生成器，支持Qwen3推理"""
    
    def __init__(self, qwen3_client: Qwen3LearnerProfileClient):
        self.qwen3_client = qwen3_client
    
    def analyze_learning_style(self, interests: list, self_description: str) -> Dict[str, str]:
        """使用Qwen3分析学习风格"""
        
        # 构建专门的学习风格分析prompt
        prompt = f"""请分析以下学习者的学习风格特征：

【学习者信息】
兴趣领域: {', '.join(interests) if interests else '未提供'}
自我描述: {self_description}

【分析要求】
请从以下三个维度分析这位学习者的学习风格特征：

1. 信息处理方式: 
   - 是否喜欢通过实践、讨论来学习 (主动型)
   - 还是更喜欢先独立思考再行动 (反思型)

2. 信息感知偏好:
   - 是否偏向具体事实和实际案例 (感知型)  
   - 还是更喜欢抽象概念和理论创新 (直觉型)

3. 信息理解模式:
   - 是否喜欢循序渐进的学习 (序列型)
   - 还是倾向于先了解全局再深入细节 (全局型)

请提供详细的分析和推理过程。"""

        try:
            # 调用Qwen3进行分析
            response = self.qwen3_client.generate(prompt)
            style_data = json.loads(response)
            
            logger.info(f"Qwen3学习风格分析完成: {style_data}")
            return style_data
            
        except Exception as e:
            logger.error(f"学习风格分析失败: {e}")
            return {
                "processing": "active",
                "perception": "sensory", 
                "understanding": "sequential"
            }

# 测试代码
if __name__ == "__main__":
    # 测试Qwen3客户端
    client = Qwen3LearnerProfileClient()
    generator = EnhancedProfileGenerator(client)
    
    # 测试学习风格分析
    test_interests = ["编程", "数据科学", "人工智能"]
    test_description = "我喜欢通过实际项目来学习，不太喜欢纯理论。遇到问题时会先自己尝试解决，然后寻求帮助。"
    
    result = generator.analyze_learning_style(test_interests, test_description)
    print("学习风格分析结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))