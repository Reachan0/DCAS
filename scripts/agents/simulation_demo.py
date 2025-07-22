#!/usr/bin/env python3
"""
模拟与反思智能体演示脚本
展示完整的数字孪生模拟和TIR反思流程
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.agents.simulation_reflection_agent import (
    SimulationReflectionAgent, SimulationConfig, CourseModule, MockLLMClient
)

def demo():
    """演示完整的模拟反思流程"""
    print("🎯 模拟与反思智能体演示")
    print("=" * 60)
    
    # 创建智能体实例
    agent = SimulationReflectionAgent(MockLLMClient(), storage_path="demo_simulation")
    
    # 场景设置：数据科学课程调整效果预测
    print("\n📊 场景：数据科学课程调整效果预测")
    print("-" * 40)
    
    # 1. 学习者画像数据
    learner_profiles = [
        {
            "user_id": "advanced_learner",
            "knowledge_mastery": {
                "Python基础": 0.9, "统计学": 0.8, "机器学习": 0.7, 
                "深度学习": 0.3, "A/B测试": 0.4, "云计算": 0.2
            },
            "learning_style": {"processing": "active", "perception": "sensory"},
            "behavioral_patterns": {"engagement_score": 0.9, "activity_level": "high"}
        },
        {
            "user_id": "intermediate_learner",
            "knowledge_mastery": {
                "Python基础": 0.7, "统计学": 0.6, "机器学习": 0.5,
                "深度学习": 0.1, "A/B测试": 0.2, "云计算": 0.1
            },
            "learning_style": {"processing": "reflective", "perception": "intuitive"},
            "behavioral_patterns": {"engagement_score": 0.7, "activity_level": "medium"}
        },
        {
            "user_id": "beginner_learner",
            "knowledge_mastery": {
                "Python基础": 0.4, "统计学": 0.3, "机器学习": 0.2,
                "深度学习": 0.0, "A/B测试": 0.1, "云计算": 0.0
            },
            "learning_style": {"processing": "active", "perception": "sensory"},
            "behavioral_patterns": {"engagement_score": 0.6, "activity_level": "low"}
        }
    ]
    
    # 2. 课程模块
    course_modules = [
        CourseModule(
            module_id="python_adv",
            title="高级Python编程",
            concepts=["高级数据结构", "性能优化", "并发编程"],
            difficulty=0.7,
            duration_hours=25.0,
            content_type="exercise"
        ),
        CourseModule(
            module_id="ml_practical",
            title="机器学习实战",
            concepts=["特征工程", "模型调优", "交叉验证"],
            difficulty=0.8,
            duration_hours=30.0,
            content_type="project"
        ),
        CourseModule(
            module_id="deep_learning",
            title="深度学习入门",
            concepts=["神经网络", "CNN", "RNN"],
            difficulty=0.9,
            duration_hours=35.0,
            content_type="case_study"
        ),
        CourseModule(
            module_id="ab_testing",
            title="A/B测试实战",
            concepts=["实验设计", "统计显著性", "结果分析"],
            difficulty=0.6,
            duration_hours=20.0,
            content_type="exercise"
        )
    ]
    
    # 3. 调整策略
    adjustment_strategies = [
        {
            "strategy_type": "add_content",
            "target_skills": ["深度学习", "A/B测试"],
            "action_items": [
                "添加深度学习实战项目",
                "增加A/B测试案例分析",
                "创建交互式实验环境"
            ]
        },
        {
            "strategy_type": "modify_content",
            "target_skills": ["机器学习"],
            "action_items": [
                "降低机器学习模块难度",
                "增加更多基础示例",
                "提供个性化学习路径"
            ]
        }
    ]
    
    # 4. 预期结果
    expected_outcomes = {
        "Python高级": 0.85,
        "机器学习": 0.80,
        "深度学习": 0.75,
        "A/B测试": 0.80,
        "整体完成率": 0.85,
        "平均参与度": 0.80
    }
    
    # 5. 运行模拟反思周期
    print("\n🚀 开始模拟反思周期...")
    result = agent.run_simulation_cycle(
        learner_profiles,
        [asdict(module) for module in course_modules],
        adjustment_strategies,
        expected_outcomes
    )
    
    # 6. 展示模拟结果
    print("\n📈 模拟结果分析")
    print("-" * 40)
    
    # 学生表现
    simulation_data = result["simulation_result"]
    print(f"\n👥 虚拟学生表现:")
    print(f"   模拟学生数量: {len(simulation_data['virtual_students'])}")
    print(f"   平均参与度: {simulation_data['engagement_metrics']['avg_engagement']:.2f}")
    print(f"   辍学率: {simulation_data['engagement_metrics']['dropout_rate']:.1%}")
    
    # 技能达成情况
    print(f"\n🎯 技能达成情况:")
    for skill, mastery in simulation_data["skill_achievements"].items():
        expected = expected_outcomes.get(skill, expected_outcomes.get("机器学习", 0.8))
        status = "✅" if mastery >= expected * 0.9 else "⚠️" if mastery >= expected * 0.7 else "❌"
        print(f"   {status} {skill}: {mastery:.2f} (预期: {expected})")
    
    # 模块完成率
    print(f"\n📊 模块完成率:")
    for module_id, rate in simulation_data["completion_rates"].items():
        module_name = next(m.title for m in course_modules if m.module_id == module_id)
        print(f"   {module_name}: {rate:.1%}")
    
    # 7. 反思洞察
    insights = result["reflection_insights"]
    print(f"\n💡 反思洞察:")
    print(f"   策略有效性: {insights['strategy_effectiveness']:.1%}")
    
    if insights["identified_issues"]:
        print(f"   发现问题 ({len(insights['identified_issues'])}个):")
        for issue in insights["identified_issues"][:3]:
            print(f"     • {issue}")
    
    print(f"\n🔧 优化建议 ({len(insights['optimization_suggestions'])}个):")
    for suggestion in insights["optimization_suggestions"][:3]:
        print(f"     • [{suggestion['urgency']}] {suggestion['suggestion']}")
    
    # 8. 下一轮迭代建议
    print(f"\n🔄 下一轮迭代重点:")
    for i, rec in enumerate(insights["next_iteration_recommendations"][:3], 1):
        print(f"   {i}. {rec}")
    
    # 9. 可视化建议
    generate_visualization_suggestions(result)
    
    # 10. 保存完整结果
    with open("simulation_cycle_demo.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 完整结果已保存: simulation_cycle_demo.json")

def generate_visualization_suggestions(result: Dict[str, Any]):
    """生成可视化建议"""
    simulation = result["simulation_result"]
    
    print(f"\n📊 可视化建议:")
    print(f"   📈 技能进步曲线: 展示各技能随时间的掌握度变化")
    print(f"   👥 学生群体分布: 按学习风格展示不同群体的表现差异")
    print(f"   ⚡ 风险预警图: 识别高风险辍学学生")
    print(f"   🎯 策略效果雷达图: 对比预期与实际达成效果")

def run_multiple_iterations():
    """运行多轮迭代优化"""
    print("\n🔄 多轮迭代优化演示")
    print("=" * 50)
    
    agent = SimulationReflectionAgent(MockLLMClient())
    
    # 初始数据
    current_outcomes = {
        "Python高级": 0.65, "机器学习": 0.55, "深度学习": 0.40,
        "A/B测试": 0.45, "整体完成率": 0.70, "平均参与度": 0.75
    }
    
    iterations = []
    
    for iteration in range(1, 4):
        print(f"\n🔄 第{iteration}轮迭代:")
        
        # 根据上一轮结果调整策略
        current_strategies = generate_iteration_strategies(current_outcomes, iteration)
        
        # 运行模拟
        result = agent.run_simulation_cycle(
            [
                {
                    "user_id": f"cohort_{iteration}",
                    "knowledge_mastery": {"基础": 0.6},
                    "learning_style": {"processing": "active"},
                    "behavioral_patterns": {"engagement_score": 0.8}
                }
            ],
            [{
                "module_id": f"module_{iteration}",
                "title": f"迭代{iteration}课程",
                "concepts": ["核心概念"],
                "difficulty": 0.7 if iteration == 1 else 0.6 if iteration == 2 else 0.5,
                "duration_hours": 25.0,
                "content_type": "exercise"
            }],
            current_strategies,
            {"核心概念": 0.8}
        )
        
        # 更新结果用于下一轮
        insights = result["reflection_insights"]
        iterations.append({
            "iteration": iteration,
            "strategy_effectiveness": insights["strategy_effectiveness"],
            "improvements": len(insights["optimization_suggestions"])
        })
        
        print(f"   策略有效性: {insights['strategy_effectiveness']:.1%}")
        print(f"   优化建议: {len(insights['optimization_suggestions'])}个")
    
    print(f"\n📊 迭代效果总结:")
    for iter_data in iterations:
        print(f"   第{iter_data['iteration']}轮: 有效性={iter_data['strategy_effectiveness']:.1%}, "
              f"优化建议={iter_data['improvements']}个")

def generate_iteration_strategies(current_outcomes: Dict[str, float], iteration: int) -> List[Dict[str, Any]]:
    """根据当前结果生成本轮优化策略"""
    strategies = []
    
    if iteration == 1:
        strategies.append({
            "strategy_type": "add_content",
            "target_skills": ["实践项目"],
            "action_items": ["增加实战案例"]
        })
    elif iteration == 2:
        strategies.append({
            "strategy_type": "modify_content",
            "target_skills": ["难度调整"],
            "action_items": ["降低整体难度"]
        })
    else:
        strategies.append({
            "strategy_type": "add_content",
            "target_skills": ["个性化"],
            "action_items": ["增加个性化路径"]
        })
    
    return strategies

if __name__ == "__main__":
    demo()
    run_multiple_iterations()