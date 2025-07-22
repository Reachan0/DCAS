#!/usr/bin/env python3
"""
学习者画像智能体 (Learner Profiling Agent)
创建、管理和动态更新学习者的全面个人档案
"""

import json
import logging
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningStyle:
    """学习风格数据结构"""
    processing: str = "active"  # active, reflective
    perception: str = "sensory"  # sensory, intuitive
    understanding: str = "sequential"  # sequential, global

@dataclass
class KnowledgeState:
    """知识状态数据结构"""
    concept_mastery: Dict[str, str] = None  # concept_id -> mastered/learning/not_started
    problem_solving_ability: str = "beginner"  # beginner, intermediate, advanced

@dataclass
class BehavioralPatterns:
    """行为模式数据结构"""
    activity_level: str = "medium"  # low, medium, high
    engagement_score: float = 0.5  # 0.0 to 1.0
    preferred_content_type: List[str] = None

@dataclass
class RawInfo:
    """原始信息数据结构"""
    interests: List[str] = None
    self_description: str = ""

@dataclass
class LearnerProfile:
    """学习者完整画像数据结构"""
    user_id: str
    last_updated: str
    knowledge_state: KnowledgeState
    learning_style: LearningStyle
    behavioral_patterns: BehavioralPatterns
    raw_info: RawInfo

class InitialProfileGenerator:
    """初始画像生成器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def _get_learning_style_prompt(self, interests: List[str], self_description: str) -> str:
        """生成学习风格推断Prompt"""
        interests_str = ", ".join(interests) if interests else "未提供"
        
        return f"""你是一位资深的教育心理学家，擅长使用菲尔德-西尔弗曼学习风格模型进行分析。

你的任务是根据以下学习者的【个人信息】，为他/她推断出最可能的学习风格。

【菲尔德-西尔弗曼学习风格模型定义】:
- **信息处理 (Processing)**: 
  - `active` (主动型): 喜欢通过实践、讨论和解释来理解。倾向于"先试再说"。
  - `reflective` (反思型): 喜欢先独立思考、消化信息。倾向于"先想后做"。
- **信息感知 (Perception)**:
  - `sensory` (感知型): 喜欢具体、有事实依据、与现实世界关联的内容。关注细节。
  - `intuitive` (直觉型): 喜欢抽象、创新的概念和理论。关注全局和可能性。
- **信息理解 (Understanding)**:
  - `sequential` (序列型): 喜欢线性、有序、一步步地学习。
  - `global` (全局型): 喜欢先了解整体框架和最终目标，再深入细节，可能会跳跃式学习。

---
【学习者个人信息】:
兴趣: {interests_str}
自我描述: {self_description}

【你的分析结果】
请严格按照以下JSON格式返回你的分析结果，不要添加任何额外的解释。

{{
  "processing": "...",
  "perception": "...",
  "understanding": "..."
}}"""

    def _infer_learning_style(self, interests: List[str], self_description: str) -> LearningStyle:
        """使用LLM推断学习风格"""
        if not self.llm_client:
            # 默认学习风格
            return LearningStyle()
        
        prompt = self._get_learning_style_prompt(interests, self_description)
        
        try:
            response = self.llm_client.generate(prompt)
            # 解析JSON响应
            style_data = json.loads(response.strip())
            return LearningStyle(**style_data)
        except Exception as e:
            logger.warning(f"学习风格推断失败，使用默认值: {e}")
            return LearningStyle()

    def create_initial_profile(self, user_id: str, initial_data: Dict[str, Any]) -> LearnerProfile:
        """创建初始学习者画像"""
        logger.info(f"为学习者 {user_id} 创建初始画像")
        
        raw_info = RawInfo(
            interests=initial_data.get('interests', []),
            self_description=initial_data.get('self_description', '')
        )
        
        # 推断学习风格
        learning_style = self._infer_learning_style(
            raw_info.interests,
            raw_info.self_description
        )
        
        # 创建完整画像
        profile = LearnerProfile(
            user_id=user_id,
            last_updated=datetime.datetime.now().isoformat(),
            knowledge_state=KnowledgeState(
                concept_mastery={},
                problem_solving_ability="beginner"
            ),
            learning_style=learning_style,
            behavioral_patterns=BehavioralPatterns(
                activity_level="medium",
                engagement_score=0.5,
                preferred_content_type=[]
            ),
            raw_info=raw_info
        )
        
        logger.info(f"初始画像创建完成: {user_id}")
        return profile

class InteractionTracker:
    """交互追踪器"""
    
    def __init__(self):
        self.events = []
    
    def track_event(self, user_id: str, event_type: str, data: Dict[str, Any]):
        """记录交互事件"""
        event = {
            "user_id": user_id,
            "event_type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data
        }
        self.events.append(event)
        logger.debug(f"记录事件: {event_type} for {user_id}")
    
    def get_user_events(self, user_id: str, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取用户的交互记录"""
        events = [e for e in self.events if e["user_id"] == user_id]
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        return events

class DynamicProfileUpdater:
    """动态画像更新器"""
    
    def __init__(self):
        self.update_rules = {
            "quiz_completed": self._update_from_quiz,
            "session_start": self._update_from_session_start,
            "session_end": self._update_from_session_end,
            "content_viewed": self._update_from_content_viewed
        }
    
    def _update_from_quiz(self, profile: LearnerProfile, event: Dict[str, Any]) -> LearnerProfile:
        """根据测验结果更新知识状态"""
        data = event["data"]
        concept_id = data.get("concept_id")
        score = data.get("score", 0)
        
        if concept_id:
            if score >= 0.8:
                mastery = "mastered"
            elif score >= 0.5:
                mastery = "learning"
            else:
                mastery = "not_started"
            
            profile.knowledge_state.concept_mastery[concept_id] = mastery
            logger.info(f"更新知识状态: {concept_id} -> {mastery}")
        
        return profile
    
    def _update_from_session_start(self, profile: LearnerProfile, event: Dict[str, Any]) -> LearnerProfile:
        """处理会话开始事件"""
        profile.behavioral_patterns.activity_level = "high"
        return profile
    
    def _update_from_session_end(self, profile: LearnerProfile, event: Dict[str, Any]) -> LearnerProfile:
        """根据会话时长更新参与度"""
        duration = event["data"].get("duration_minutes", 0)
        
        # 更新参与度分数
        if duration >= 60:
            engagement = min(profile.behavioral_patterns.engagement_score + 0.1, 1.0)
        elif duration >= 30:
            engagement = min(profile.behavioral_patterns.engagement_score + 0.05, 1.0)
        else:
            engagement = max(profile.behavioral_patterns.engagement_score - 0.05, 0.0)
        
        profile.behavioral_patterns.engagement_score = engagement
        logger.info(f"更新参与度: {engagement}")
        return profile
    
    def _update_from_content_viewed(self, profile: LearnerProfile, event: Dict[str, Any]) -> LearnerProfile:
        """根据内容查看更新偏好"""
        content_type = event["data"].get("content_type")
        if content_type:
            if content_type not in profile.behavioral_patterns.preferred_content_type:
                profile.behavioral_patterns.preferred_content_type.append(content_type)
                # 限制最多5种偏好类型
                profile.behavioral_patterns.preferred_content_type = profile.behavioral_patterns.preferred_content_type[-5:]
            logger.info(f"更新内容偏好: {content_type}")
        
        return profile
    
    def update_profile(self, profile: LearnerProfile, events: List[Dict[str, Any]]) -> LearnerProfile:
        """根据事件列表更新画像"""
        for event in events:
            event_type = event["event_type"]
            if event_type in self.update_rules:
                profile = self.update_rules[event_type](profile, event)
            else:
                logger.debug(f"未处理的事件类型: {event_type}")
        
        # 更新最后更新时间
        profile.last_updated = datetime.datetime.now().isoformat()
        logger.info(f"画像更新完成: {profile.user_id}")
        return profile

class LearnerProfilingAgent:
    """学习者画像智能体主类"""
    
    def __init__(self, llm_client=None, storage_path: str = "learner_profiles"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.initial_generator = InitialProfileGenerator(llm_client)
        self.tracker = InteractionTracker()
        self.updater = DynamicProfileUpdater()
    
    def create_profile(self, user_id: str, initial_data: Dict[str, Any]) -> str:
        """创建新学习者画像"""
        profile = self.initial_generator.create_initial_profile(user_id, initial_data)
        self._save_profile(profile)
        return profile.user_id
    
    def track_interaction(self, user_id: str, event_type: str, data: Dict[str, Any]):
        """记录交互事件"""
        self.tracker.track_event(user_id, event_type, data)
    
    def update_profile(self, user_id: str) -> LearnerProfile:
        """根据交互记录更新画像"""
        profile = self._load_profile(user_id)
        events = self.tracker.get_user_events(user_id)
        
        if events:
            profile = self.updater.update_profile(profile, events)
            self._save_profile(profile)
        
        return profile
    
    def get_profile(self, user_id: str) -> LearnerProfile:
        """获取学习者画像"""
        return self._load_profile(user_id)
    
    def _save_profile(self, profile: LearnerProfile):
        """保存画像到文件"""
        file_path = self.storage_path / f"{profile.user_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
    
    def _load_profile(self, user_id: str) -> LearnerProfile:
        """从文件加载画像"""
        file_path = self.storage_path / f"{user_id}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"学习者画像不存在: {user_id}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 反序列化数据
        return LearnerProfile(
            user_id=data["user_id"],
            last_updated=data["last_updated"],
            knowledge_state=KnowledgeState(**data["knowledge_state"]),
            learning_style=LearningStyle(**data["learning_style"]),
            behavioral_patterns=BehavioralPatterns(**data["behavioral_patterns"]),
            raw_info=RawInfo(**data["raw_info"])
        )

class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def generate(self, prompt: str) -> str:
        """模拟生成学习风格"""
        if "喜欢动手实践" in prompt:
            return '{"processing": "active", "perception": "sensory", "understanding": "sequential"}'
        elif "喜欢独立思考" in prompt:
            return '{"processing": "reflective", "perception": "intuitive", "understanding": "global"}'
        else:
            return '{"processing": "active", "perception": "sensory", "understanding": "sequential"}'

# 测试代码
if __name__ == "__main__":
    # 创建测试实例
    mock_llm = MockLLMClient()
    agent = LearnerProfilingAgent(mock_llm)
    
    # 测试创建画像
    user_id = "test_user_001"
    initial_data = {
        "interests": ["编程", "数学", "游戏设计"],
        "self_description": "我是一个喜欢动手实践的人，不太喜欢纯理论，更愿意通过实际项目来学习新知识"
    }
    
    # 创建初始画像
    agent.create_profile(user_id, initial_data)
    
    # 模拟交互
    agent.track_interaction(user_id, "quiz_completed", {
        "concept_id": "Python基础语法",
        "score": 0.85
    })
    
    agent.track_interaction(user_id, "session_end", {
        "duration_minutes": 45
    })
    
    agent.track_interaction(user_id, "content_viewed", {
        "content_type": "case_study"
    })
    
    # 更新画像
    updated_profile = agent.update_profile(user_id)
    
    # 打印结果
    print("学习者画像:")
    print(json.dumps(asdict(updated_profile), ensure_ascii=False, indent=2))