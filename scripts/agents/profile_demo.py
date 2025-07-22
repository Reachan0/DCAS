#!/usr/bin/env python3
"""
学习者画像智能体演示脚本
展示完整的创建、交互记录和更新流程
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.agents.learner_profiling_agent import LearnerProfilingAgent, MockLLMClient

def demo():
    """演示完整流程"""
    print("🎯 学习者画像智能体演示")
    print("=" * 50)
    
    # 创建智能体实例
    llm_client = MockLLMClient()
    agent = LearnerProfilingAgent(llm_client, storage_path="demo_profiles")
    
    # 1. 创建新学习者画像
    print("\n1️⃣ 创建新学习者画像...")
    user_id = "demo_user_001"
    initial_data = {
        "interests": ["人工智能", "数据科学", "机器学习", "Python编程"],
        "self_description": "我是一个喜欢动手实践的人，不太喜欢纯理论。我喜欢通过实际项目来学习新知识，遇到问题时喜欢先尝试解决再寻求帮助。"
    }
    
    agent.create_profile(user_id, initial_data)
    print(f"✅ 画像创建完成: {user_id}")
    
    # 2. 获取初始画像
    print("\n2️⃣ 初始画像:")
    initial_profile = agent.get_profile(user_id)
    print(json.dumps({
        "learning_style": {
            "processing": initial_profile.learning_style.processing,
            "perception": initial_profile.learning_style.perception,
            "understanding": initial_profile.learning_style.understanding
        },
        "knowledge_state": initial_profile.knowledge_state.__dict__,
        "behavioral_patterns": initial_profile.behavioral_patterns.__dict__
    }, ensure_ascii=False, indent=2))
    
    # 3. 模拟学习交互
    print("\n3️⃣ 模拟学习交互...")
    
    # 模拟测验完成
    agent.track_interaction(user_id, "quiz_completed", {
        "concept_id": "Python基础语法",
        "score": 0.9,
        "difficulty": "medium"
    })
    
    agent.track_interaction(user_id, "quiz_completed", {
        "concept_id": "面向对象编程",
        "score": 0.7,
        "difficulty": "hard"
    })
    
    # 模拟学习会话
    agent.track_interaction(user_id, "session_start", {
        "session_id": "session_001",
        "timestamp": "2024-07-19T10:00:00"
    })
    
    agent.track_interaction(user_id, "session_end", {
        "session_id": "session_001",
        "duration_minutes": 60,
        "content_types": ["video", "exercise", "quiz"]
    })
    
    # 模拟内容查看
    agent.track_interaction(user_id, "content_viewed", {
        "content_type": "case_study",
        "duration_minutes": 15
    })
    
    agent.track_interaction(user_id, "content_viewed", {
        "content_type": "interactive_exercise",
        "duration_minutes": 25
    })
    
    print("✅ 交互记录完成")
    
    # 4. 更新画像
    print("\n4️⃣ 更新学习者画像...")
    updated_profile = agent.update_profile(user_id)
    print("✅ 画像更新完成")
    
    # 5. 显示更新后的画像
    print("\n5️⃣ 更新后的画像:")
    result = {
        "user_id": updated_profile.user_id,
        "last_updated": updated_profile.last_updated,
        "learning_style": {
            "processing": updated_profile.learning_style.processing,
            "perception": updated_profile.learning_style.perception,
            "understanding": updated_profile.learning_style.understanding
        },
        "knowledge_state": {
            "concept_mastery": updated_profile.knowledge_state.concept_mastery,
            "problem_solving_ability": updated_profile.knowledge_state.problem_solving_ability
        },
        "behavioral_patterns": {
            "activity_level": updated_profile.behavioral_patterns.activity_level,
            "engagement_score": updated_profile.behavioral_patterns.engagement_score,
            "preferred_content_type": updated_profile.behavioral_patterns.preferred_content_type
        }
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 6. 个性化建议
    print("\n6️⃣ 个性化学习建议:")
    suggestions = generate_learning_suggestions(updated_profile)
    for suggestion in suggestions:
        print(f"  • {suggestion}")
    
    print("\n🎉 演示完成！")

def generate_learning_suggestions(profile) -> list:
    """基于画像生成个性化建议"""
    suggestions = []
    
    # 基于学习风格
    if profile.learning_style.processing == "active":
        suggestions.append("建议多参与实践项目和讨论")
    else:
        suggestions.append("建议先独立思考再参与讨论")
    
    if profile.learning_style.perception == "sensory":
        suggestions.append("推荐具体的案例和实际项目")
    else:
        suggestions.append("推荐抽象概念和理论框架")
    
    # 基于知识状态
    mastered_concepts = [k for k, v in profile.knowledge_state.concept_mastery.items() if v == "mastered"]
    if mastered_concepts:
        suggestions.append(f"可以继续学习: {', '.join(list(profile.knowledge_state.concept_mastery.keys())[:3])}")
    
    # 基于行为模式
    if profile.behavioral_patterns.engagement_score > 0.7:
        suggestions.append("保持当前的高参与度学习状态")
    
    if profile.behavioral_patterns.preferred_content_type:
        suggestions.append(f"推荐内容类型: {', '.join(profile.behavioral_patterns.preferred_content_type)}")
    
    return suggestions

if __name__ == "__main__":
    demo()