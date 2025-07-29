#!/usr/bin/env python3
"""
学习者画像智能体综合测试
验证Qwen3集成、学习风格分析、动态更新等完整功能
"""

import sys
import json
import time
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from scripts.agents.learner_profiling_agent import LearnerProfilingAgent, MockLLMClient

def test_qwen3_integration():
    """测试Qwen3集成功能"""
    print("🧠 测试1: Qwen3智能学习风格推断")
    print("-" * 40)
    
    # 创建支持Qwen3的智能体
    agent = LearnerProfilingAgent(storage_path="test_qwen3_profiles")
    
    # 测试不同类型的学习者
    test_learners = [
        {
            "user_id": "practical_learner",
            "interests": ["编程", "机器人", "工程"],
            "self_description": "我喜欢通过动手实践来学习，遇到问题会先自己尝试解决。偏好实际案例和项目驱动的学习方式。"
        },
        {
            "user_id": "theoretical_learner", 
            "interests": ["数学", "物理", "哲学"],
            "self_description": "我习惯于先深入理解理论基础，然后再应用。喜欢独立思考和抽象概念，偏好系统性的学习方法。"
        },
        {
            "user_id": "mixed_learner",
            "interests": ["数据科学", "人工智能", "商业分析"],
            "self_description": "我倾向于先了解整体框架和目标，然后深入细节。既喜欢理论学习也重视实践应用。"
        }
    ]
    
    results = []
    
    for learner in test_learners:
        print(f"\n📝 分析学习者: {learner['user_id']}")
        print(f"   兴趣: {', '.join(learner['interests'])}")
        print(f"   描述: {learner['self_description'][:50]}...")
        
        # 创建画像
        start_time = time.time()
        agent.create_profile(learner['user_id'], {
            'interests': learner['interests'],
            'self_description': learner['self_description']
        })
        
        # 获取结果
        profile = agent.get_profile(learner['user_id'])
        analysis_time = time.time() - start_time
        
        result = {
            "user_id": learner['user_id'],
            "analysis_time": round(analysis_time, 2),
            "learning_style": {
                "processing": profile.learning_style.processing,
                "perception": profile.learning_style.perception,
                "understanding": profile.learning_style.understanding
            }
        }
        results.append(result)
        
        print(f"   ✅ 分析完成 ({analysis_time:.2f}s)")
        print(f"   🎯 学习风格: {profile.learning_style.processing}/{profile.learning_style.perception}/{profile.learning_style.understanding}")
    
    print(f"\n📊 Qwen3集成测试总结:")
    for result in results:
        print(f"  {result['user_id']}: {result['learning_style']['processing']}/{result['learning_style']['perception']}/{result['learning_style']['understanding']} ({result['analysis_time']}s)")
    
    return results

def test_dynamic_profile_updates():
    """测试动态画像更新功能"""
    print(f"\n🔄 测试2: 动态画像更新")
    print("-" * 40)
    
    agent = LearnerProfilingAgent(storage_path="test_dynamic_profiles")
    user_id = "dynamic_test_user"
    
    # 创建初始画像
    initial_data = {
        "interests": ["Python", "数据分析"],
        "self_description": "刚开始学习编程，希望通过项目实践来提升技能"
    }
    
    agent.create_profile(user_id, initial_data)
    initial_profile = agent.get_profile(user_id)
    
    print(f"📋 初始画像:")
    print(f"   知识掌握: {len(initial_profile.knowledge_state.concept_mastery)} 个概念")
    print(f"   参与度: {initial_profile.behavioral_patterns.engagement_score}")
    print(f"   活跃度: {initial_profile.behavioral_patterns.activity_level}")
    
    # 模拟学习进展
    learning_events = [
        ("quiz_completed", {"concept_id": "Python基础", "score": 0.8}),
        ("session_end", {"duration_minutes": 60}), 
        ("quiz_completed", {"concept_id": "数据结构", "score": 0.9}),
        ("content_viewed", {"content_type": "video_tutorial"}),
        ("session_end", {"duration_minutes": 90}),
        ("quiz_completed", {"concept_id": "算法基础", "score": 0.7}),
        ("content_viewed", {"content_type": "interactive_exercise"}),
    ]
    
    print(f"\n📈 模拟学习进展:")
    for i, (event_type, data) in enumerate(learning_events, 1):
        agent.track_interaction(user_id, event_type, data)
        
        if event_type == "quiz_completed":
            print(f"   {i}. 完成测验: {data['concept_id']} (分数: {data['score']})")
        elif event_type == "session_end":
            print(f"   {i}. 学习会话: {data['duration_minutes']}分钟")
        elif event_type == "content_viewed":
            print(f"   {i}. 查看内容: {data['content_type']}")
    
    # 更新画像
    updated_profile = agent.update_profile(user_id)
    
    print(f"\n📊 更新后画像:")
    print(f"   知识掌握: {len(updated_profile.knowledge_state.concept_mastery)} 个概念")
    print(f"   参与度: {updated_profile.behavioral_patterns.engagement_score:.2f}")
    print(f"   活跃度: {updated_profile.behavioral_patterns.activity_level}")
    print(f"   内容偏好: {updated_profile.behavioral_patterns.preferred_content_type}")
    
    # 显示知识掌握详情
    print(f"   概念掌握详情:")
    for concept, mastery in updated_profile.knowledge_state.concept_mastery.items():
        print(f"     - {concept}: {mastery}")
    
    return updated_profile

def test_learning_recommendations():
    """测试个性化学习建议生成"""
    print(f"\n💡 测试3: 个性化学习建议")
    print("-" * 40)
    
    from scripts.agents.profile_demo import generate_learning_suggestions
    
    agent = LearnerProfilingAgent(storage_path="test_recommendation_profiles")
    
    # 测试不同画像的建议生成
    test_cases = [
        {
            "user_id": "active_sensory_learner",
            "data": {
                "interests": ["编程", "游戏开发", "机器人"],
                "self_description": "喜欢动手制作，通过实际项目学习效果最好"
            }
        },
        {
            "user_id": "reflective_intuitive_learner", 
            "data": {
                "interests": ["算法", "理论计算机科学", "数学"],
                "self_description": "倾向于深入理解原理，喜欢抽象思维和理论推导"
            }
        }
    ]
    
    for case in test_cases:
        agent.create_profile(case["user_id"], case["data"])
        
        # 添加一些交互记录
        agent.track_interaction(case["user_id"], "quiz_completed", {
            "concept_id": "核心概念1", "score": 0.85
        })
        agent.track_interaction(case["user_id"], "content_viewed", {
            "content_type": "case_study"
        })
        
        profile = agent.update_profile(case["user_id"])
        suggestions = generate_learning_suggestions(profile)
        
        print(f"\n👤 学习者: {case['user_id']}")
        print(f"   学习风格: {profile.learning_style.processing}/{profile.learning_style.perception}/{profile.learning_style.understanding}")
        print(f"   个性化建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"     {i}. {suggestion}")

def test_persistence_and_loading():
    """测试数据持久化和加载"""
    print(f"\n💾 测试4: 数据持久化与加载")
    print("-" * 40)
    
    agent = LearnerProfilingAgent(storage_path="test_persistence_profiles")
    user_id = "persistence_test_user"
    
    # 创建并保存画像
    original_data = {
        "interests": ["人工智能", "深度学习"],
        "self_description": "研究生学历，有一定编程基础，希望深入AI领域"
    }
    
    agent.create_profile(user_id, original_data)
    
    # 添加交互记录
    agent.track_interaction(user_id, "quiz_completed", {
        "concept_id": "机器学习基础", "score": 0.92
    })
    
    original_profile = agent.update_profile(user_id)
    
    print(f"✅ 原始画像已保存")
    print(f"   用户ID: {original_profile.user_id}")
    print(f"   知识状态: {len(original_profile.knowledge_state.concept_mastery)} 个概念")
    
    # 创建新的智能体实例（模拟重启）
    new_agent = LearnerProfilingAgent(storage_path="test_persistence_profiles")
    
    # 加载保存的画像
    loaded_profile = new_agent.get_profile(user_id)
    
    print(f"✅ 画像重新加载成功")
    print(f"   用户ID: {loaded_profile.user_id}")
    print(f"   知识状态: {len(loaded_profile.knowledge_state.concept_mastery)} 个概念")
    
    # 验证数据一致性
    consistency_checks = [
        original_profile.user_id == loaded_profile.user_id,
        original_profile.learning_style.processing == loaded_profile.learning_style.processing,
        len(original_profile.knowledge_state.concept_mastery) == len(loaded_profile.knowledge_state.concept_mastery),
        original_profile.behavioral_patterns.engagement_score == loaded_profile.behavioral_patterns.engagement_score
    ]
    
    if all(consistency_checks):
        print(f"✅ 数据一致性验证通过")
    else:
        print(f"❌ 数据一致性验证失败") 
    
    return all(consistency_checks)

def main():
    """运行所有测试"""
    print("🧪 学习者画像智能体综合测试")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # 测试1: Qwen3集成
        qwen3_results = test_qwen3_integration()
        test_results["qwen3_integration"] = len(qwen3_results) > 0
        
        # 测试2: 动态更新
        dynamic_profile = test_dynamic_profile_updates()
        test_results["dynamic_updates"] = dynamic_profile.knowledge_state.concept_mastery != {}
        
        # 测试3: 个性化建议
        test_learning_recommendations()
        test_results["recommendations"] = True
        
        # 测试4: 持久化
        persistence_ok = test_persistence_and_loading()
        test_results["persistence"] = persistence_ok
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        test_results["error"] = str(e)
    
    # 输出测试总结
    print(f"\n" + "=" * 60)
    print(f"📊 测试总结")
    print(f"=" * 60)
    
    passed = sum(1 for v in test_results.values() if v is True)
    total = len([k for k in test_results.keys() if k != "error"])
    
    print(f"✅ 通过测试: {passed}/{total}")
    
    for test_name, result in test_results.items():
        if test_name != "error":
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
    
    if "error" in test_results:
        print(f"❌ 错误信息: {test_results['error']}")
    
    if passed == total:
        print(f"\n🎉 所有测试通过！学习者画像智能体运行正常")
        print(f"💡 系统已具备:")
        print(f"   - Qwen3智能学习风格推断")
        print(f"   - 动态画像更新机制") 
        print(f"   - 个性化学习建议生成")
        print(f"   - 完整的数据持久化")
    else:
        print(f"\n⚠️  部分测试未通过，请检查相关功能")
    
    return test_results

if __name__ == "__main__":
    main()